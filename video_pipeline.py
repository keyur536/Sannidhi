"""
video_pipeline.py  -  Video-Based Student Enrollment Pipeline
==============================================================
Processes student videos from Videos/ folder.

Drop videos named like:  Keyur_Chaudhari.mp4  into  Attendance/Videos/

Run:
    python video_pipeline.py

Pipeline flow:
    1. Load MTCNN + FaceNet from model.py
    2. Load Excel roster (PGCP-AI.xlsx) for PRN lookup
    3. For each video in Videos/:
       a. Extract 1 frame every 10th frame (max 80 frames)
       b. MTCNN crops the largest face from each frame
       c. 13x augmentation (original, flip, brightness, blur, rotations ±15/30/45/60°)
       d. FaceNet embeds each variant → average into master 512-d vector
       e. INSERT into attendance.db
    4. Archive processed video → Processed_Videos/
"""




import json
import math
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageEnhance, ImageFilter

from model import load_models_enrollment, embed_face, FACE_MARGIN, MIN_FACE_CONF, FACENET_TRANSFORM

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR          = Path(__file__).parent
VIDEOS_DIR        = BASE_DIR / "Videos"
PROCESSED_VID_DIR = BASE_DIR / "Processed_Videos"
DB_PATH           = BASE_DIR / "attendance.db"
EXCEL_PATH        = BASE_DIR / "PGCP-AI.xlsx"

EXCEL_NAME_COL    = "Name"
EXCEL_PRN_COL     = "PRN No."
SUPPORTED_VID     = {".mp4", ".avi", ".mov", ".mkv"}

# Frame sampling
FRAME_STEP        = 10       # extract 1 frame every N frames
MAX_FRAMES        = 80       # cap per video

# ── Augmentation list (13x) ───────────────────────────────────────────────────
def _rotate(deg: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        return img.rotate(deg, expand=True, resample=Image.BICUBIC)
    return fn

AUGMENTATIONS: list[tuple[str, Callable[[Image.Image], Image.Image]]] = [
    ("original",    lambda img: img),
    ("hflip",       lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
    ("rot_p15",     _rotate(+15)),
    ("rot_p30",     _rotate(+30)),
    ("rot_p45",     _rotate(+45)),
    ("rot_p60",     _rotate(+60)),
    ("rot_m15",     _rotate(-15)),
    ("rot_m30",     _rotate(-30)),
    ("rot_m45",     _rotate(-45)),
    ("rot_m60",     _rotate(-60)),
    ("bright_up",   lambda img: ImageEnhance.Brightness(img).enhance(1.30)),
    ("bright_dn",   lambda img: ImageEnhance.Brightness(img).enhance(0.70)),
    ("blur",        lambda img: img.filter(ImageFilter.GaussianBlur(radius=1))),
]


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
W = 68

def banner(title):   print(f"\n{'=' * W}\n  {title}\n{'=' * W}")
def section(title):  print(f"\n  {'─' * (W - 2)}\n  {title}\n  {'─' * (W - 2)}")
def ok(msg):   print(f"    [  OK  ]  {msg}")
def skip(msg): print(f"    [ SKIP ]  {msg}")
def warn(msg): print(f"    [ WARN ]  {msg}")
def info(msg): print(f"    [ INFO ]  {msg}")
def fail(msg): print(f"    [ FAIL ]  {msg}")
def dbmsg(msg):print(f"    [  DB  ]  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            prn       TEXT PRIMARY KEY,
            name      TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def get_enrolled_prns(conn: sqlite3.Connection) -> set:
    return {r[0] for r in conn.execute("SELECT prn FROM students").fetchall()}

def insert_student(conn: sqlite3.Connection, prn: str, name: str, emb: np.ndarray):
    conn.execute(
        "INSERT OR REPLACE INTO students (prn, name, embedding) VALUES (?, ?, ?)",
        (prn, name, json.dumps(emb.tolist())),
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
#  EXCEL LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
def _norm(name: str) -> str:
    return " ".join(name.lower().split())

def load_prn_map(excel_path: Path) -> dict:
    df = pd.read_excel(excel_path, dtype={EXCEL_PRN_COL: str})
    return {
        _norm(str(row[EXCEL_NAME_COL]).strip()): (
            str(row[EXCEL_NAME_COL]).strip(),
            str(row[EXCEL_PRN_COL]).strip(),
        )
        for _, row in df.iterrows()
    }

def lookup(folder_name: str, prn_map: dict):
    return prn_map.get(_norm(folder_name))


# ══════════════════════════════════════════════════════════════════════════════
#  NAME PARSING FROM FILENAME
# ══════════════════════════════════════════════════════════════════════════════
def filename_to_name(video_path: Path) -> str:
    """Keyur_Chaudhari.mp4  ->  'Keyur Chaudhari'"""
    stem = video_path.stem.replace("_", " ").strip()
    return " ".join(w.capitalize() for w in stem.split())


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_frames(video_path: Path) -> list[np.ndarray]:
    """Sample 1 frame every FRAME_STEP, up to MAX_FRAMES. Returns BGR arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path.name}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = total / fps if fps > 0 else 0
    info(f"  {total} frames | {fps:.1f} fps | {dur:.1f}s")
    info(f"  Sampling every {FRAME_STEP} frames (cap: {MAX_FRAMES} frames)")

    frames = []
    frame_idx = 0
    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_STEP == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    info(f"  Extracted {len(frames)} frames for processing")
    return frames


# ══════════════════════════════════════════════════════════════════════════════
#  MTCNN FACE CROPPING (per frame)
# ══════════════════════════════════════════════════════════════════════════════
def crop_face_from_frame(bgr_frame: np.ndarray, mtcnn: MTCNN) -> Image.Image | None:
    """Detect & crop the largest face in a BGR frame. Returns PIL crop or None."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(pil)
    if boxes is None or len(boxes) == 0:
        return None

    conf = float(probs[0]) if probs is not None else 0.0
    if conf < MIN_FACE_CONF:
        return None

    x1, y1, x2, y2 = [int(v) for v in boxes[0]]
    w, h = pil.size
    face = pil.crop((
        max(0, x1 - FACE_MARGIN), max(0, y1 - FACE_MARGIN),
        min(w, x2 + FACE_MARGIN), min(h, y2 + FACE_MARGIN),
    ))
    return face


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION + EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_master_embedding(
    crops: list[Image.Image],
    facenet: InceptionResnetV1,
    device: torch.device,
) -> np.ndarray | None:
    """Apply 13x augmentations to every crop, embed, return L2-normalised mean vector."""
    n_aug = len(AUGMENTATIONS)
    total_expected = len(crops) * n_aug
    info(f"  {len(crops)} crop(s) x {n_aug} aug(s) = {total_expected} embedding vectors")

    all_vecs: list[np.ndarray] = []
    for crop in crops:
        for label, aug_fn in AUGMENTATIONS:
            try:
                augmented = aug_fn(crop)
                vec = embed_face(augmented, facenet, device)
                all_vecs.append(vec)
            except Exception as e:
                warn(f"  Aug '{label}' failed: {e}")

    if not all_vecs:
        return None

    info(f"  {len(all_vecs)} vectors collected. Averaging...")
    master = np.mean(all_vecs, axis=0)
    return master / (np.linalg.norm(master) + 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
#  ARCHIVE
# ══════════════════════════════════════════════════════════════════════════════
def archive_video(src: Path, archive_root: Path):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = archive_root / f"{src.stem}__{ts}{src.suffix}"
    shutil.move(str(src), str(dst))
    ok(f"Archived -> Processed_Videos/{dst.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION  (importable by other scripts)
# ══════════════════════════════════════════════════════════════════════════════
def run_video_pipeline(
    videos_dir:    Path,
    db_path:       Path,
    excel_path:    Path,
    processed_dir: Path,
    mtcnn:         MTCNN,
    facenet:       InceptionResnetV1,
    device:        torch.device,
    prn_map:       dict,
    enrolled_prns: set,
) -> dict:
    """Process all videos in videos_dir. Returns stats dict."""
    processed_dir.mkdir(exist_ok=True)

    videos = sorted(
        v for v in videos_dir.iterdir()
        if v.is_file() and v.suffix.lower() in SUPPORTED_VID
    )

    if not videos:
        info(f"No videos found in {videos_dir}/")
        return {"added": [], "skipped_db": [], "skipped_excel": [], "failed": []}

    info(f"Found {len(videos)} video(s) in {videos_dir.name}/")
    conn = open_db(db_path)

    stats = {"added": [], "skipped_db": [], "skipped_excel": [], "failed": []}
    total = len(videos)

    for idx, video_path in enumerate(videos, 1):
        student_name = filename_to_name(video_path)
        section(f"[{idx}/{total}]  {video_path.name}  ->  '{student_name}'")

        # PRN lookup
        result = lookup(student_name, prn_map)
        if result is None:
            warn(f"Not found in Excel roster. Check spelling: '{student_name}'")
            stats["skipped_excel"].append(student_name)
            continue
        canonical_name, prn = result
        info(f"Excel -> PRN: {prn}  |  Name: {canonical_name}")

        if prn in enrolled_prns:
            skip(f"PRN {prn} already enrolled. Skipping.")
            stats["skipped_db"].append(canonical_name)
            continue

        # Extract frames
        try:
            frames = extract_frames(video_path)
        except IOError as e:
            fail(str(e))
            stats["failed"].append(student_name)
            continue

        if not frames:
            fail("No frames extracted.")
            stats["failed"].append(student_name)
            continue

        # MTCNN crop each frame
        crops: list[Image.Image] = []
        skipped_count = 0
        for frame in frames:
            crop = crop_face_from_frame(frame, mtcnn)
            if crop is not None:
                crops.append(crop)
            else:
                skipped_count += 1

        ok(f"{len(crops)} face crop(s) from {len(frames)} frames "
           f"({skipped_count} frame(s) had no confident face)")

        if not crops:
            fail("Zero face crops obtained. Check video quality / lighting.")
            stats["failed"].append(student_name)
            continue

        # Augment + embed
        master = generate_master_embedding(crops, facenet, device)
        if master is None:
            fail("Embedding generation failed.")
            stats["failed"].append(student_name)
            continue

        ok(f"Master vector: 512-d | norm={np.linalg.norm(master):.4f}")

        # Save to DB
        insert_student(conn, prn, canonical_name, master)
        enrolled_prns.add(prn)
        dbmsg(f"Inserted  PRN={prn}  Name='{canonical_name}'")
        ok(f"'{canonical_name}' enrolled in attendance.db")
        stats["added"].append(canonical_name)

        # Archive video
        archive_video(video_path, processed_dir)

    conn.close()
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN  (standalone usage)
# ══════════════════════════════════════════════════════════════════════════════
def main():
    banner("Video-Based Student Enrollment Pipeline")
    print(f"  Videos folder  : {VIDEOS_DIR}")
    print(f"  Database       : {DB_PATH}")
    print(f"  Excel roster   : {EXCEL_PATH}")
    print(f"  Frame sampling : every {FRAME_STEP} frames (max {MAX_FRAMES})")
    print(f"  Augmentations  : {len(AUGMENTATIONS)}x per frame")

    for d in [VIDEOS_DIR, PROCESSED_VID_DIR]:
        d.mkdir(exist_ok=True)

    if not EXCEL_PATH.exists():
        print(f"\n[ERROR] Excel not found: {EXCEL_PATH}")
        sys.exit(1)

    # Check for videos before loading heavy models
    videos = [v for v in VIDEOS_DIR.iterdir()
               if v.is_file() and v.suffix.lower() in SUPPORTED_VID]
    if not videos:
        print(f"\n  [ INFO ]  No videos in {VIDEOS_DIR}/")
        print(f"            Drop .mp4/.avi files named like  Keyur_Chaudhari.mp4  and re-run.")
        sys.exit(0)

    # Load models from model.py
    section("Loading Models")
    mtcnn, facenet, device = load_models_enrollment()
    ok(f"MTCNN + FaceNet ready.  Device: {device}")

    info("Loading Excel roster...")
    prn_map = load_prn_map(EXCEL_PATH)
    ok(f"{len(prn_map)} students in roster.")

    conn = open_db(DB_PATH)
    enrolled = get_enrolled_prns(conn)
    conn.close()
    ok(f"DB ready. Currently enrolled: {len(enrolled)} student(s).")

    # Run
    stats = run_video_pipeline(
        videos_dir    = VIDEOS_DIR,
        db_path       = DB_PATH,
        excel_path    = EXCEL_PATH,
        processed_dir = PROCESSED_VID_DIR,
        mtcnn         = mtcnn,
        facenet       = facenet,
        device        = device,
        prn_map       = prn_map,
        enrolled_prns = enrolled,
    )

    # Report
    banner("VIDEO PIPELINE COMPLETE  -  Final Report")
    print(f"  {'Newly enrolled':<28}: {len(stats['added'])}")
    print(f"  {'Already in DB (skipped)':<28}: {len(stats['skipped_db'])}")
    print(f"  {'Not in Excel (skipped)':<28}: {len(stats['skipped_excel'])}")
    print(f"  {'Failed':<28}: {len(stats['failed'])}")

    if stats["added"]:
        print("\n  ENROLLED")
        for n in stats["added"]: print(f"    +  {n}")
    if stats["skipped_db"]:
        print("\n  ALREADY ENROLLED")
        for n in stats["skipped_db"]: print(f"    -  {n}")
    if stats["skipped_excel"]:
        print("\n  NOT IN EXCEL (check spelling)")
        for n in stats["skipped_excel"]: print(f"    ?  {n}")
    if stats["failed"]:
        print("\n  FAILED")
        for n in stats["failed"]: print(f"    x  {n}")

    # Final DB state
    conn = sqlite3.connect(DB_PATH)
    all_s = conn.execute("SELECT prn, name FROM students ORDER BY name").fetchall()
    conn.close()
    print(f"\n  ENROLLED STUDENTS ({len(all_s)} total)")
    print(f"  {'PRN':<18} Name")
    print(f"  {'-'*18} {'-'*35}")
    for prn, name in all_s:
        print(f"  {prn:<18} {name}")
    print()


if __name__ == "__main__":
    main()
