"""
model.py  -  Centralised Model Loader & Inference Helpers
=========================================================
Houses all deep-learning model initialisation and inference
utilities used across the attendance system.

Models loaded here:
    • MTCNN         — Multi-face detection & alignment
    • FaceNet       — InceptionResnetV1 / VGGFace2 embedding
    • GFPGAN v1.3   — GAN-based face restoration (optional)

Import usage:
    from model import (
        load_models, load_gfpgan, enhance_with_gfpgan,
        detect_and_crop_faces, embed_face, cosine_similarity,
        match_faces_to_students, FACENET_TRANSFORM,
        COSINE_THRESHOLD, MIN_FACE_CONF, FACE_MARGIN,
        GFPGAN_MODEL_PATH, GFPGAN_UPSCALE, GFPGAN_AVAILABLE,
    )
"""

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms

# GFPGAN — optional, gracefully disabled if not installed
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "attendance.db"

# ── Config ─────────────────────────────────────────────────────────────────────
COSINE_THRESHOLD = 0.60          # match accepted only above this score
FACE_MARGIN      = 20            # px added around MTCNN bounding boxes
MIN_FACE_CONF    = 0.90          # MTCNN minimum detection confidence

# ── GFPGAN config ──────────────────────────────────────────────────────────────
GFPGAN_MODEL_PATH = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
GFPGAN_UPSCALE    = 4

# ── FaceNet preprocessing ──────────────────────────────────────────────────────
FACENET_TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ══════════════════════════════════════════════════════════════════════════════
#  Model Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    """Load MTCNN (face detector) and FaceNet (embedder). Returns (mtcnn, facenet, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(
        image_size=160,
        margin=FACE_MARGIN,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        keep_all=True,       # detect ALL faces in the classroom photo
        device=device,
    )
    facenet = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
    return mtcnn, facenet, device


def load_models_enrollment():
    """Load MTCNN (single-face mode for enrollment) and FaceNet. Returns (mtcnn, facenet, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(
        image_size=160,
        margin=FACE_MARGIN,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        keep_all=False,          # single face per image for enrollment
        select_largest=True,     # pick the biggest face (closest to camera)
        device=device,
    )
    facenet = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
    return mtcnn, facenet, device


def load_gfpgan():
    """
    Load GFPGANer once.
    Returns the restorer object, or None if GFPGAN is not installed.
    """
    if not GFPGAN_AVAILABLE:
        return None
    try:
        restorer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=GFPGAN_UPSCALE,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,   # face pixels only — faster
        )
        return restorer
    except Exception:
        return None


def enhance_with_gfpgan(
    pil_crop: Image.Image,
    restorer,
) -> Image.Image:
    """
    Enhance a single PIL face crop using GFPGAN.
    Converts PIL (RGB) -> BGR numpy -> GFPGAN -> RGB PIL.
    Falls back to the original crop if enhancement fails.
    """
    try:
        bgr = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
        _, _, restored_bgr = restorer.enhance(
            bgr,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
        )
        if restored_bgr is None:
            return pil_crop
        return Image.fromarray(cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        return pil_crop


# ══════════════════════════════════════════════════════════════════════════════
#  Inference Helpers
# ══════════════════════════════════════════════════════════════════════════════

def detect_and_crop_faces(
    pil_img: Image.Image,
    mtcnn: MTCNN,
) -> list[tuple[Image.Image, list[int]]]:
    """
    Run MTCNN on the full classroom image.
    Returns a list of (cropped_PIL_image, [x1,y1,x2,y2]) tuples,
    one entry per detected face (confidence >= MIN_FACE_CONF).
    """
    boxes, probs = mtcnn.detect(pil_img)
    if boxes is None:
        return []

    results = []
    w, h = pil_img.size
    for box, prob in zip(boxes, probs):
        if prob is None or float(prob) < MIN_FACE_CONF:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1 - FACE_MARGIN)
        y1 = max(0, y1 - FACE_MARGIN)
        x2 = min(w, x2 + FACE_MARGIN)
        y2 = min(h, y2 + FACE_MARGIN)
        crop = pil_img.crop((x1, y1, x2, y2))
        results.append((crop, [x1, y1, x2, y2]))
    return results


@torch.no_grad()
def embed_face(
    pil_crop: Image.Image,
    facenet:  InceptionResnetV1,
    device:   torch.device,
) -> np.ndarray:
    """Convert a PIL crop to a L2-normalised 512-d FaceNet embedding."""
    tensor = FACENET_TRANSFORM(pil_crop.convert("RGB")).unsqueeze(0).to(device)
    emb    = facenet(tensor).squeeze(0).cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-10)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # both are L2-normalised, so dot == cosine


def match_faces_to_students(
    face_crops:   list[tuple[Image.Image, list[int]]],
    db_records:   list[dict],
    facenet:      InceptionResnetV1,
    device:       torch.device,
    threshold:    float = COSINE_THRESHOLD,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    For each detected face, find the best-matching student in the DB.
    A match is accepted only if cosine similarity >= threshold.

    Returns:
        present  - list of {student_id, name, prn, score, crop_img}
        absent   - list of {student_id, name, prn}  (enrolled students NOT matched)
        unknown  - list of {crop_img, best_score}  (detected faces not in DB)
    """
    if not db_records:
        unknown = [{"crop_img": c, "best_score": 0.0} for c, _ in face_crops]
        return [], [], unknown

    matched_prns: set[str] = set()
    present: list[dict] = []
    unknown: list[dict] = []

    for crop_img, _ in face_crops:
        live_emb = embed_face(crop_img, facenet, device)

        best_score = -1.0
        best_rec   = None
        for rec in db_records:
            if rec["prn"] in matched_prns:
                continue
            score = cosine_similarity(live_emb, rec["embedding"])
            if score > best_score:
                best_score = score
                best_rec   = rec

        if best_rec is not None and best_score >= threshold:
            matched_prns.add(best_rec["prn"])
            present.append({
                "student_id": best_rec["student_id"],
                "name":     best_rec["name"],
                "prn":      best_rec["prn"],
                "score":    best_score,
                "crop_img": crop_img,
            })
        else:
            unknown.append({
                "crop_img":   crop_img,
                "best_score": best_score,
            })

    absent = [
        {"student_id": r["student_id"], "name": r["name"], "prn": r["prn"]}
        for r in db_records
        if r["prn"] not in matched_prns
    ]

    return present, absent, unknown
