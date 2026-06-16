"""
pipeline.py  -  Video-Only Student Enrollment Pipeline
=======================================================
ONE script. ONE command. Handles student enrollment via VIDEOS ONLY.

Drop new student VIDEOS into:  Videos/Student_Name.mp4

Run:
    python pipeline.py

The script automatically:
    STAGE 1  Load models & database once

    VIDEO PIPELINE (Videos/)
    STAGE 2  Extract frames (every 10th, max 80)
    STAGE 3  MTCNN crop each frame
    STAGE 4  Augment (13x: rotations +-15/30/45/60, flip, brightness, blur)
    STAGE 5  FaceNet embed all -> master vector
    STAGE 6  Insert into attendance.db
    STAGE 7  Archive -> Processed_Videos/

Requirements:
    pip install facenet-pytorch torch torchvision Pillow pandas openpyxl opencv-python
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Import video pipeline (shares models — no double loading)
from video_pipeline import run_video_pipeline, PROCESSED_VID_DIR

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# BASE_DIR      = Path(__file__).parent
from pathlib import Path

BASE_DIR = Path(r"C:\Users\DELL\Desktop\Projects\Attendance_Using_facenet")

VIDEOS_DIR    = BASE_DIR / "Videos"             # Admin drops videos here
DB_PATH       = BASE_DIR / "attendance.db"
EXCEL_PATH    = BASE_DIR / "PGCP-AI.xlsx"

EXCEL_NAME_COL = "Name"
EXCEL_PRN_COL  = "PRN No."

# MTCNN
FACE_MARGIN   = 20
MIN_CONF      = 0.90


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

W = 66

def banner(title: str) -> None:
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")

def stage(n: int, title: str) -> None:
    print(f"\n  {'─' * (W-2)}")
    print(f"  STAGE {n}  |  {title}")
    print(f"  {'─' * (W-2)}")

def ok(msg: str)   -> None: print(f"    [  OK  ]  {msg}")
def skip(msg: str) -> None: print(f"    [ SKIP ]  {msg}")
def warn(msg: str) -> None: print(f"    [ WARN ]  {msg}")
def info(msg: str) -> None: print(f"    [ INFO ]  {msg}")
def fail(msg: str) -> None: print(f"    [ FAIL ]  {msg}")
def db(msg: str)   -> None: print(f"    [  DB  ]  {msg}")


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

def get_enrolled_prns(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT prn FROM students").fetchall()}


# ══════════════════════════════════════════════════════════════════════════════
#  EXCEL LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def _norm(name: str) -> str:
    return " ".join(name.lower().split())

def load_prn_map(excel_path: Path) -> dict[str, tuple[str, str]]:
    df = pd.read_excel(excel_path, dtype={EXCEL_PRN_COL: str})
    missing = {EXCEL_NAME_COL, EXCEL_PRN_COL} - set(df.columns)
    if missing:
        print(f"[ERROR] Missing Excel columns: {missing}")
        sys.exit(1)
    return {
        _norm(str(row[EXCEL_NAME_COL]).strip()): (
            str(row[EXCEL_NAME_COL]).strip(),
            str(row[EXCEL_PRN_COL]).strip(),
        )
        for _, row in df.iterrows()
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    banner("Video-Only Student Enrollment Pipeline")
    now = datetime.now().strftime("%A, %d %B %Y  |  %H:%M:%S")
    print(f"  Started       : {now}")
    print(f"  Video input   : {VIDEOS_DIR}")
    print(f"  Database      : {DB_PATH}")
    print(f"  Excel         : {EXCEL_PATH}")
    print(f"  Video aug     : 13x per frame (rotations +-15/30/45/60, flip, brightness, blur)")

    # Create required folders
    VIDEOS_DIR.mkdir(exist_ok=True)

    if not EXCEL_PATH.exists():
        print(f"\n[ERROR] Excel file not found: {EXCEL_PATH}")
        sys.exit(1)

    # Check if there are any videos to process
    new_videos = sorted(
        v for v in VIDEOS_DIR.iterdir()
        if v.is_file() and v.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )

    if not new_videos:
        print(f"\n  [ INFO ]  Nothing to process.")
        print(f"            Drop video files into Videos/ (e.g. Student_Name.mp4) and re-run.")
        sys.exit(0)

    info(f"Found {len(new_videos)} video(s) to process.")

    # ── Load models ONCE ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Compute device : {device}")

    stage(1, "Loading Models & Database")
    info("Loading MTCNN...")
    mtcnn = MTCNN(
        image_size=160, margin=FACE_MARGIN, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        keep_all=False, select_largest=True, device=device,
    )
    ok("MTCNN ready.")

    info("Loading FaceNet (InceptionResnetV1 / vggface2)...")
    facenet = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
    ok("FaceNet ready.")

    info("Loading Excel roster...")
    prn_map = load_prn_map(EXCEL_PATH)
    ok(f"Loaded {len(prn_map)} records from {EXCEL_PATH.name}")

    info("Opening attendance.db...")
    conn = open_db(DB_PATH)
    enrolled = get_enrolled_prns(conn)
    conn.close()
    ok(f"Database ready. Currently enrolled: {len(enrolled)} student(s).")

    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO PIPELINE  (Videos/)
    # ══════════════════════════════════════════════════════════════════════════
    banner(f"VIDEO PIPELINE  ({len(new_videos)} video(s))")
    vid_stats = run_video_pipeline(
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

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════════
    banner("PIPELINE COMPLETE  -  Final Report")

    total_added   = len(vid_stats["added"])
    total_skip    = len(vid_stats["skipped_db"])
    total_noexcel = len(vid_stats["skipped_excel"])
    total_failed  = len(vid_stats["failed"])

    print(f"  {'Source':<20} {'Added':>6} {'Skipped':>8} {'No Excel':>9} {'Failed':>7}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*9} {'-'*7}")
    print(f"  {'Videos (Videos/)':<20} {total_added:>6} "
          f"{total_skip:>8} {total_noexcel:>9} {total_failed:>7}")

    if vid_stats["added"]:
        print(f"\n  NEWLY ENROLLED")
        for n in vid_stats["added"]:
            print(f"    +  {n}")

    if vid_stats["failed"]:
        print(f"\n  FAILED (retake videos)")
        for n in vid_stats["failed"]:
            print(f"    x  {n}")

    if vid_stats["skipped_excel"]:
        print(f"\n  NOT IN EXCEL (check spelling)")
        for n in vid_stats["skipped_excel"]:
            print(f"    ?  {n}")

    # Final DB state
    conn = sqlite3.connect(DB_PATH)
    all_students = conn.execute(
        "SELECT prn, name FROM students ORDER BY name"
    ).fetchall()
    conn.close()
    print(f"\n  FULL ENROLLMENT LIST  ({len(all_students)} students total)")
    print(f"  {'PRN':<18} Name")
    print(f"  {'-'*18} {'-'*35}")
    for prn, name in all_students:
        print(f"  {prn:<18} {name}")
    print()


if __name__ == "__main__":
    main()
