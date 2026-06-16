"""
process_videos.py  -  Batch Video Processing Pipeline (OpenCV only)
====================================================================
Uses ONLY OpenCV — no PyTorch, no MTCNN, no extra downloads.
Face detection is done with OpenCV's bundled Haar Cascades
(haarcascade_frontalface_default + haarcascade_profileface) which
ship inside the opencv-python package itself.

Workflow:
  1. Drop videos named like  Keyur_Chaudhari.mp4  into Attendance/Videos/
  2. Run:  python process_videos.py
  3. Then: python update_database.py   (to embed the new crops)

Usage:
    python process_videos.py

Requirements:
    pip install opencv-python Pillow
"""

import sys
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
VIDEOS_DIR  = BASE_DIR / "Videos"
CROPPED_DIR = BASE_DIR / "Cropped_DataSet"

# Locate OpenCV's bundled data directory (where cascade XMLs live)
CV2_DATA_DIR = Path(os.path.dirname(cv2.__file__)) / "data"

# ── Settings ───────────────────────────────────────────────────────────────────
SUPPORTED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
SUPPORTED_IMG_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Positions in the video to sample (as a fraction of total frames)
FRAME_POSITIONS  = [0.15, 0.35, 0.50, 0.70, 0.85]
TARGET_CROPS     = len(FRAME_POSITIONS)   # 5

# Skip a student's video if their folder already has this many crops
MIN_EXISTING_CROPS = 5

# Haar cascade tuning
FACE_MARGIN       = 30    # px added around the detected bounding box
SCALE_FACTOR      = 1.10  # how much the image size is reduced at each scale
MIN_NEIGHBOURS    = 5     # higher = fewer false positives
MIN_FACE_SIZE     = (60, 60)  # ignore tiny detections (pixels)


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(msg: str)  -> None: print(f"\n{'=' * 65}\n  {msg}\n{'=' * 65}")
def _section(msg: str) -> None: print(f"\n  {'-' * 60}\n  {msg}\n  {'-' * 60}")
def _ok(msg: str)      -> None: print(f"  [ OK  ]  {msg}")
def _skip(msg: str)    -> None: print(f"  [SKIP ]  {msg}")
def _warn(msg: str)    -> None: print(f"  [WARN ]  {msg}")
def _info(msg: str)    -> None: print(f"  [INFO ]  {msg}")
def _fail(msg: str)    -> None: print(f"  [FAIL ]  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  OpenCV face detector  (Haar Cascade — no downloads, ships with opencv-python)
# ══════════════════════════════════════════════════════════════════════════════

def load_cascades() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    """
    Load the two bundled Haar cascades:
      - frontal face  (person looking straight at camera)
      - profile face  (person slightly turned left or right)
    Using both together dramatically improves recall on angled shots.
    """
    frontal_path = str(CV2_DATA_DIR / "haarcascade_frontalface_default.xml")
    profile_path = str(CV2_DATA_DIR / "haarcascade_profileface.xml")

    frontal = cv2.CascadeClassifier(frontal_path)
    profile  = cv2.CascadeClassifier(profile_path)

    if frontal.empty() or profile.empty():
        print(f"[ERROR] Could not load Haar cascade files from: {CV2_DATA_DIR}")
        sys.exit(1)

    return frontal, profile


def detect_face_opencv(
    bgr_frame: np.ndarray,
    frontal:   cv2.CascadeClassifier,
    profile:   cv2.CascadeClassifier,
) -> tuple[int, int, int, int] | None:
    """
    Run both cascades on a BGR frame.
    Returns the (x, y, w, h) of the LARGEST detected face, or None.

    Strategy:
      1. Try frontal cascade first (fastest, most common case).
      2. If no result, try profile cascade (handles turned heads).
      3. If multiple faces found, pick the largest one by area.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)   # boost contrast for dim frames

    def _detect(cascade: cv2.CascadeClassifier) -> list[tuple]:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBOURS,
            minSize=MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return list(faces) if len(faces) > 0 else []

    faces = _detect(frontal)

    # Also try mirrored frame for profile to catch both left & right turns
    if not faces:
        faces = _detect(profile)
    if not faces:
        mirrored = cv2.flip(gray, 1)
        cv2.equalizeHist(mirrored, mirrored)
        flipped_faces = cv2.CascadeClassifier(
            str(CV2_DATA_DIR / "haarcascade_profileface.xml")
        ).detectMultiScale(
            mirrored,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBOURS,
            minSize=MIN_FACE_SIZE,
        )
        if len(flipped_faces) > 0:
            # Un-mirror the x coordinate
            h_img, w_img = gray.shape
            faces = [(w_img - x - w, y, w, h) for (x, y, w, h) in flipped_faces]

    if not faces:
        return None

    # Pick the largest face by area
    largest = max(faces, key=lambda r: r[2] * r[3])
    return tuple(largest)   # (x, y, w, h)


# ══════════════════════════════════════════════════════════════════════════════
#  Name parsing
# ══════════════════════════════════════════════════════════════════════════════

def filename_to_name(video_path: Path) -> str:
    """
    Keyur_Chaudhari.mp4  ->  'Keyur Chaudhari'
    Aditi_Vikas_Deo.avi  ->  'Aditi Vikas Deo'
    """
    stem = video_path.stem.replace("_", " ").strip()
    return " ".join(w.capitalize() for w in stem.split())


# ══════════════════════════════════════════════════════════════════════════════
#  Existing-crop check
# ══════════════════════════════════════════════════════════════════════════════

def count_existing_crops(student_name: str) -> int:
    folder = CROPPED_DIR / student_name
    if not folder.exists():
        return 0
    return sum(
        1 for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMG_EXT
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Frame extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(
    video_path: Path,
    positions:  list[float],
) -> list[tuple[np.ndarray, float]]:
    """
    Open the video and seek to each fractional position.
    Returns a list of (BGR_ndarray, position_fraction).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path.name}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_s   = total_frames / fps if fps > 0 else 0
    _info(f"  {total_frames} frames  |  {fps:.1f} fps  |  ~{duration_s:.1f}s")

    results = []
    for pos in positions:
        target = max(0, min(int(total_frames * pos), total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            _warn(f"  Could not read frame at {pos:.0%} (frame #{target})")
            continue
        results.append((frame, pos))

    cap.release()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Crop and save
# ══════════════════════════════════════════════════════════════════════════════

def crop_and_save(
    bgr_frame: np.ndarray,
    pos:       float,
    out_dir:   Path,
    frame_idx: int,
    frontal:   cv2.CascadeClassifier,
    profile:   cv2.CascadeClassifier,
) -> bool:
    """
    Detect a face in bgr_frame, apply margin, crop, and save as JPEG.
    Returns True on success.
    """
    result = detect_face_opencv(bgr_frame, frontal, profile)

    if result is None:
        _warn(f"    Frame @{pos:.0%}  ->  No face detected.")
        return False

    x, y, w, h = result
    img_h, img_w = bgr_frame.shape[:2]

    # Apply margin (clamped to image boundaries)
    x1 = max(0, x - FACE_MARGIN)
    y1 = max(0, y - FACE_MARGIN)
    x2 = min(img_w, x + w + FACE_MARGIN)
    y2 = min(img_h, y + h + FACE_MARGIN)

    face_bgr  = bgr_frame[y1:y2, x1:x2]
    face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil  = Image.fromarray(face_rgb)

    out_path = out_dir / f"frame_{frame_idx:02d}_pos{int(pos*100):03d}.jpg"
    face_pil.save(out_path, format="JPEG", quality=95)
    _ok(f"    Frame @{pos:.0%}  ->  Saved: {out_path.name}  "
        f"(face {w}x{h}px, margin+{FACE_MARGIN}px)")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Per-video processor
# ══════════════════════════════════════════════════════════════════════════════

def process_video(
    video_path:   Path,
    student_name: str,
    frontal:      cv2.CascadeClassifier,
    profile:      cv2.CascadeClassifier,
) -> tuple[int, int]:
    """
    Full pipeline for one video.
    Returns (crops_saved, frames_attempted).
    """
    out_dir = CROPPED_DIR / student_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        frames = extract_frames(video_path, FRAME_POSITIONS)
    except IOError as exc:
        _fail(str(exc))
        return 0, 0

    if not frames:
        _fail("  No readable frames could be extracted.")
        return 0, 0

    crops_saved = 0
    for idx, (frame, pos) in enumerate(frames, start=1):
        if crop_and_save(frame, pos, out_dir, idx, frontal, profile):
            crops_saved += 1

    return crops_saved, len(frames)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _banner("Batch Video Processing Pipeline  (OpenCV only)")
    print(f"  Videos folder    : {VIDEOS_DIR}")
    print(f"  Output folder    : {CROPPED_DIR}")
    print(f"  Detector         : OpenCV Haar Cascade (frontal + profile)")
    print(f"  Frames per video : {TARGET_CROPS}  "
          f"({', '.join(f'{int(p*100)}%' for p in FRAME_POSITIONS)})")
    print(f"  Skip threshold   : >= {MIN_EXISTING_CROPS} existing crops")

    VIDEOS_DIR.mkdir(exist_ok=True)
    CROPPED_DIR.mkdir(exist_ok=True)

    # ── Collect videos ─────────────────────────────────────────────────────
    videos = sorted(
        v for v in VIDEOS_DIR.iterdir()
        if v.is_file() and v.suffix.lower() in SUPPORTED_VIDEO_EXT
    )

    if not videos:
        print(f"\n  [INFO ] No video files found in {VIDEOS_DIR}")
        print("          Drop .mp4 / .avi files named like  Keyur_Chaudhari.mp4  and re-run.")
        sys.exit(0)

    _info(f"Found {len(videos)} video file(s).")

    # ── Load cascades ───────────────────────────────────────────────────────
    _info("Loading OpenCV Haar Cascades...")
    frontal, profile = load_cascades()
    _ok("Cascades loaded (no internet required).\n")

    # ── Process each video ──────────────────────────────────────────────────
    results = {"skipped": [], "success": [], "partial": [], "failed": []}

    for video_path in videos:
        student_name = filename_to_name(video_path)
        _section(f"{video_path.name}  ->  '{student_name}'")

        existing = count_existing_crops(student_name)
        if existing >= MIN_EXISTING_CROPS:
            _skip(
                f"Already has {existing} cropped image(s) in Cropped_DataSet/ "
                f"(threshold: {MIN_EXISTING_CROPS}). Skipping."
            )
            results["skipped"].append(student_name)
            continue

        if existing > 0:
            _info(f"  Found {existing} existing crop(s) — will try to add more.")

        crops_saved, frames_tried = process_video(
            video_path, student_name, frontal, profile
        )

        if crops_saved == 0:
            _fail(f"  0 / {frames_tried} crops saved for '{student_name}'.")
            results["failed"].append(student_name)
        elif crops_saved < TARGET_CROPS:
            _warn(f"  Partial: {crops_saved} / {frames_tried} crops saved.")
            results["partial"].append((student_name, crops_saved))
        else:
            _ok(f"  {crops_saved} / {frames_tried} crops saved.")
            results["success"].append((student_name, crops_saved))

    # ── Final report ────────────────────────────────────────────────────────
    _banner("BATCH COMPLETE  -  Final Report")
    total = len(videos)
    print(f"  Total videos     : {total}")
    print(f"  Fully succeeded  : {len(results['success'])}")
    print(f"  Partial crops    : {len(results['partial'])}")
    print(f"  Failed (0 crops) : {len(results['failed'])}")
    print(f"  Skipped          : {len(results['skipped'])}")

    if results["success"]:
        print("\n  SUCCESS")
        for name, n in results["success"]:
            print(f"    + {name:<40} ({n} crop(s))")

    if results["partial"]:
        print("\n  PARTIAL  (fewer than 5 crops — consider retaking the video)")
        for name, n in results["partial"]:
            print(f"    ~ {name:<40} ({n} crop(s))")

    if results["failed"]:
        print("\n  FAILED  (0 crops — faces not detected)")
        for name in results["failed"]:
            print(f"    x {name}")
        print("\n  Tips for failed videos:")
        print("    - Ensure the student faces the camera directly")
        print("    - Use good lighting (avoid strong backlight)")
        print("    - Try lowering MIN_NEIGHBOURS in the script (currently "
              f"{MIN_NEIGHBOURS})")

    if results["skipped"]:
        print("\n  SKIPPED  (already enrolled)")
        for name in results["skipped"]:
            print(f"    - {name}")

    print(f"\n  Next step: run  python update_database.py  to embed new students.")
    print()


if __name__ == "__main__":
    main()
