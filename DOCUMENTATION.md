# AI-Powered Smart Classroom Attendance System
### Complete Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [AI Models Used](#3-ai-models-used)
4. [Technology Stack](#4-technology-stack)
5. [Project File Structure](#5-project-file-structure)
6. [Installation & Setup](#6-installation--setup)
7. [Phase 1 — Student Enrollment Pipeline](#7-phase-1--student-enrollment-pipeline)
8. [Phase 2 — Attendance Dashboard](#8-phase-2--attendance-dashboard)
9. [Database Schema](#9-database-schema)
10. [Data Augmentation Strategy](#10-data-augmentation-strategy)
11. [Configuration Reference](#11-configuration-reference)
12. [How Matching Works](#12-how-matching-works)
13. [Troubleshooting](#13-troubleshooting)
14. [Resume Summary](#14-resume-summary)

---

## 1. Project Overview

### What Is This?
This is an **end-to-end AI-based biometric attendance system** that uses **Computer Vision** and **Deep Learning** to automatically detect and recognize multiple students from a single classroom photograph — completely eliminating manual roll calls.

### The Core Idea
A teacher uploads **one photo** of the classroom. The system:
- Detects every face in the photo using **MTCNN**
- Compares each face against stored student data using **FaceNet**
- Instantly produces a **Present / Absent / Unknown** list

No student needs to do anything. No RFID cards. No fingerprints. One photo is enough.

### Two-Phase Architecture
The system works in two completely separate phases:

| Phase | Name | When It Runs | Who Runs It |
|-------|------|-------------|------------|
| **Phase 1** | Student Enrollment | Once per student | Admin / Teacher |
| **Phase 2** | Daily Attendance | Every class session | Admin / Teacher |

---

## 2. System Architecture

### Phase 1 — Enrollment Pipeline Flow

```
Student Video (.mp4)
        │
        ▼
[ OpenCV Frame Extraction ]
  Every 10th frame → max 80 frames
        │
        ▼
[ MTCNN Face Detection ]
  Detect largest face in each frame
  Crop with 20px margin (160×160 px)
  Discard if confidence < 90%
        │
        ▼
[ 13× Data Augmentation ]
  Original, Flip, Blur,
  Brightness ±, Rotation ±15/30/45/60°
        │
        ▼
[ FaceNet (InceptionResnetV1) ]
  Generate 512-d embedding per variant
        │
        ▼
[ Average All Embeddings ]
  Mean of all vectors → L2 Normalized
  → 1 Master 512-d Vector per student
        │
        ▼
[ SQLite Database (attendance.db) ]
  INSERT (PRN, Name, Embedding JSON)
        │
        ▼
[ Archive Video ]
  Move to Processed_Videos/ with timestamp
```

### Phase 2 — Attendance Pipeline Flow

```
Classroom Photo (upload)
        │
        ▼
[ MTCNN — Multi-Face Mode ]
  Detect ALL faces in one pass
  keep_all=True → N face crops
        │
        ▼
[ GFPGAN v1.3 (Optional) ]
  Restore & sharpen blurry crops
  4× super-resolution upscaling
        │
        ▼
[ FaceNet Embedding ]
  Generate 512-d vector per live face
        │
        ▼
[ Cosine Similarity Matching ]
  Compare vs every enrolled student
  Threshold: ≥ 0.60 (60%) to match
  Each student matched at most once
        │
        ▼
[ Results ]
  ✅ Present  — matched above threshold
  ❌ Absent   — enrolled but not detected
  ⚠️ Unknown  — detected but no match
        │
        ▼
[ Streamlit Dashboard ]
  Display results with face crops,
  confidence scores, stat pills
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No CNN Classifier** | Using embedding + Cosine Similarity means new students can be added without retraining any model |
| **Video over Photos** | Videos yield up to 80 frames and multiple face angles, producing a more robust master embedding |
| **13× Augmentation** | Artificially increases training diversity without collecting more data |
| **GAN for Enhancement Only** | GFPGAN only sharpens live crops — never generates synthetic training data |
| **Modular Codebase** | All AI logic is in `model.py`; `app.py` and `pipeline.py` are kept clean |
| **SQLite** | Zero-configuration, fully portable, no server needed |

---

## 3. AI Models Used

### MTCNN — Multi-task Cascaded Convolutional Neural Network

| Property | Value |
|----------|-------|
| **Full Name** | Multi-task Cascaded CNN |
| **Library** | `facenet-pytorch` |
| **Task** | Face Detection + Alignment |
| **Pretrained On** | WIDER FACE Dataset |
| **Output** | Bounding Boxes + Detection Probabilities |

**How it works:** MTCNN runs three cascaded CNNs (P-Net, R-Net, O-Net) in sequence. Each stage progressively refines the bounding boxes. It can detect multiple faces simultaneously with precise alignment.

**In this project:**
- **Enrollment mode:** `keep_all=False`, `select_largest=True` — only crops the biggest (closest) face per frame
- **Attendance mode:** `keep_all=True` — detects ALL faces in the classroom photo

---

### FaceNet — InceptionResnetV1

| Property | Value |
|----------|-------|
| **Full Name** | InceptionResnetV1 |
| **Library** | `facenet-pytorch` |
| **Task** | Facial Embedding Generation |
| **Pretrained On** | VGGFace2 (3.3 million images, 9131 identities) |
| **Output** | 512-dimensional L2-normalized vector |
| **Training Objective** | Triplet Loss / Metric Learning |

**How it works:** FaceNet maps every face image to a point in a 512-dimensional space. Faces of the same person cluster together; faces of different people are far apart. **No retraining needed** — new students are simply added by generating and storing their embedding.

**In this project:**
- Input: 160×160 px RGB face crop (normalized: mean=0.5, std=0.5)
- Output: 512-d float32 vector, L2-normalized to unit length
- Used for both **enrollment** (generate master vector) and **attendance** (generate live vector)

---

### GFPGAN v1.3 — Generative Face Restoration

| Property | Value |
|----------|-------|
| **Full Name** | Generative Facial Prior GAN |
| **Library** | `gfpgan` |
| **Task** | Face Restoration & Super-Resolution |
| **Pretrained On** | FFHQ Dataset (70,000 high-quality faces) |
| **Output** | Restored high-resolution face image |
| **Upscale Factor** | 4× (configurable) |

**How it works:** GFPGAN uses a pre-trained GAN that has learned the prior distribution of clean, high-resolution faces. It uses this prior to hallucinate fine details (skin texture, eyes, hair) from degraded inputs.

**In this project:** Optional enhancement step between MTCNN cropping and FaceNet embedding. Enables accurate recognition even from wide-angle or blurry classroom photos.

---

## 4. Technology Stack

### Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` + `torchvision` | ≥ 2.0 | Deep learning framework — all model inference |
| `facenet-pytorch` | 2.6.0 | Pre-built MTCNN and InceptionResnetV1 |
| `opencv-python` (cv2) | ≥ 4.8 | Video frame extraction, BGR↔RGB conversion |
| `streamlit` | ≥ 1.35 | Admin web dashboard UI |
| `Pillow` (PIL) | ≥ 10.0 | Image loading, cropping, augmentation |
| `numpy` | ≥ 1.24 | Embedding vector math, array operations |
| `pandas` + `openpyxl` | ≥ 2.0 | Excel roster loading, student PRN lookup |
| `sqlite3` | stdlib | Lightweight database (no installation needed) |
| `gfpgan` | 1.3.8 | GAN face restoration (optional) |

**One-liner Platform Statement (for resume):**
> Python Libraries — PyTorch, OpenCV, Streamlit, SQLite, Pandas, NumPy, facenet-pytorch, GFPGAN

---

## 5. Project File Structure

```
Attendance_Using_facenet/
│
├── pipeline.py          ← Master enrollment orchestrator (RUN THIS FIRST)
├── video_pipeline.py    ← Worker: frame extract → MTCNN → augment → FaceNet → DB
├── app.py               ← Streamlit admin dashboard (RUN FOR ATTENDANCE)
├── model.py             ← Centralized AI loader (imported, never run directly)
├── process_videos.py    ← Lightweight OpenCV-only frame extractor (optional)
├── generate_docs.py     ← PDF documentation generator
│
├── PGCP-AI.xlsx         ← Student roster (Name + PRN No. columns)
├── attendance.db        ← SQLite database (auto-created on first enrollment)
├── README.md            ← GitHub README
├── DOCUMENTATION.md     ← This file
│
├── Videos/              ← DROP STUDENT VIDEOS HERE before running pipeline.py
└── Processed_Videos/    ← Videos auto-moved here after enrollment (with timestamp)
```

### File Responsibilities

#### `pipeline.py` — Enrollment Orchestrator
- The **main entry point** for student enrollment
- Loads MTCNN and FaceNet models **once** (shared across all videos — efficient)
- Reads `PGCP-AI.xlsx` to build Name → PRN mapping
- Creates/opens `attendance.db`
- Calls `run_video_pipeline()` from `video_pipeline.py` with all models pre-loaded
- Prints a final enrollment report (Added / Skipped / Failed)

#### `video_pipeline.py` — Enrollment Worker
- Contains the actual video processing logic
- `extract_frames()` — samples 1 frame every 10 frames, max 80 per video
- `crop_face_from_frame()` — runs MTCNN on each frame, crops the face
- `generate_master_embedding()` — applies 13× augmentations, runs FaceNet, averages
- `insert_student()` — writes PRN + Name + embedding JSON to SQLite
- `archive_video()` — moves processed video to `Processed_Videos/`
- Can also be run standalone as `python video_pipeline.py`

#### `model.py` — AI Model Central Hub
- Defines all model loading functions: `load_models()`, `load_models_enrollment()`, `load_gfpgan()`
- Defines all inference helpers: `detect_and_crop_faces()`, `embed_face()`, `cosine_similarity()`, `match_faces_to_students()`
- Defines `load_database()` for reading embeddings from SQLite
- **Never run directly** — always imported by `app.py` and `video_pipeline.py`
- All constants live here: `COSINE_THRESHOLD`, `FACE_MARGIN`, `MIN_FACE_CONF`

#### `app.py` — Streamlit Admin Dashboard
- Password-protected login gate (`admin123`)
- File uploader for classroom photos
- Calls `detect_and_crop_faces()` → optional GFPGAN → `match_faces_to_students()`
- Renders Present / Absent / Unknown sections with face crops and confidence %
- Sidebar shows DB stats, system info, and GFPGAN toggle

---

## 6. Installation & Setup

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU (CUDA) | Optional | NVIDIA GPU (faster embedding) |
| Disk Space | 2 GB | 5 GB (for model weight cache) |
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/Attendance_Using_facenet.git
cd Attendance_Using_facenet
```

### Step 2 — Install Core Dependencies
```bash
pip install torch torchvision facenet-pytorch opencv-python streamlit pandas openpyxl Pillow numpy
```

> **GPU Acceleration:** For NVIDIA GPUs, install PyTorch with CUDA from https://pytorch.org/get-started/locally/ instead of the command above.

### Step 3 — Install GFPGAN (Optional)
```bash
pip install gfpgan
```
GFPGAN is optional. Without it, the system still works perfectly — GFPGAN just improves accuracy on blurry photos.

### Step 4 — Prepare the Excel File
Open `PGCP-AI.xlsx` and ensure it has exactly these two column headers:

| Name | PRN No. |
|------|---------|
| Keyur Chaudhari | 2200010101 |
| Sannidhi Patel | 2200010102 |

> **Important:** Column headers must be exactly `Name` and `PRN No.` (case-sensitive, space-sensitive).

---

## 7. Phase 1 — Student Enrollment Pipeline

### How to Enroll Students

**Step 1:** Place student video files in the `Videos/` folder.
- File naming format: Replace spaces with underscores, use student's full name
- Example: `Keyur_Chaudhari.mp4`, `Sannidhi_Patel.mp4`
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`
- Ideal video: 5–15 seconds, student looking at the camera, good lighting

**Step 2:** Make sure the student's name is in `PGCP-AI.xlsx`

**Step 3:** Run the pipeline:
```bash
python pipeline.py
```

### What the Pipeline Does (Step by Step)

| Stage | Action | Detail |
|-------|--------|--------|
| **Stage 1** | Load Models | Load MTCNN + FaceNet once (shared for all videos) |
| **Stage 2** | Read Excel | Build Name → PRN lookup dictionary |
| **Stage 3** | Open DB | Create `attendance.db` if not exists |
| **Stage 4** | Frame Extraction | Sample 1 frame every 10 frames, max 80 frames per video |
| **Stage 5** | MTCNN Cropping | Detect largest face per frame, crop with 20px margin |
| **Stage 6** | Augmentation | Apply 13× variants to each crop |
| **Stage 7** | FaceNet Embedding | Generate 512-d vector for every augmented crop |
| **Stage 8** | Average | Average all vectors → 1 L2-normalized master vector |
| **Stage 9** | Database Insert | `INSERT OR REPLACE INTO students (prn, name, embedding)` |
| **Stage 10** | Archive | Move video to `Processed_Videos/` with timestamp |

### Expected Terminal Output
```
==================================================================
  Video-Only Student Enrollment Pipeline
==================================================================
  Started : Monday, 16 June 2026 | 23:45:00
  VIDEO PIPELINE  (2 video(s))
  ──────────────────────────────────────────
  [1/2]  Keyur_Chaudhari.mp4  ->  'Keyur Chaudhari'
    [ INFO ]  Excel -> PRN: 2200010101  |  Name: Keyur Chaudhari
    [ INFO ]  250 frames | 30.0 fps | 8.3s
    [ INFO ]  Extracted 25 frames for processing
    [  OK  ]  22 face crop(s) from 25 frames (3 frames had no confident face)
    [ INFO ]  22 crop(s) x 13 aug(s) = 286 embedding vectors
    [  OK  ]  Master vector: 512-d | norm=1.0000
    [  DB  ]  Inserted  PRN=2200010101  Name='Keyur Chaudhari'
    [  OK  ]  'Keyur Chaudhari' enrolled in attendance.db
    [  OK  ]  Archived -> Processed_Videos/Keyur_Chaudhari__20260616_234512.mp4
```

### Re-Enrolling a Student
Run `pipeline.py` again with a new video. `INSERT OR REPLACE` will **update** the embedding, not create a duplicate.

---

## 8. Phase 2 — Attendance Dashboard

### How to Launch
```bash
streamlit run app.py
```
Open your browser at: **http://localhost:8501**

### Login
- **Password:** `admin123` (change in `app.py` → `APP_PASSWORD`)

### Dashboard Walkthrough

**1. Upload Photo**
Drag and drop any classroom photograph (JPG, PNG, BMP, WEBP).

**2. Face Detection**
MTCNN runs automatically and detects all faces. A success message shows how many faces were detected.

**3. Enhancement (Optional)**
Toggle "Enable GFPGAN Enhancement" in the sidebar. Each detected face crop is passed through GFPGAN before FaceNet embedding. Recommended for blurry or wide-angle photos.

**4. Matching**
FaceNet embeds each detected face. Cosine Similarity is calculated against every enrolled student. A student is marked **Present** if their best score is ≥ 60%.

**5. Results Display**

| Section | Color | Content |
|---------|-------|---------|
| ✅ Present Students | Green | Name, PRN, Confidence %, face crop |
| ❌ Absent Students | Red | Name, PRN of enrolled students with no match |
| ⚠️ Unknown Faces | Orange | Face crops that didn't match any enrolled student |

**6. Stats Bar**
Live pills at the top show: Enrolled Total / Present / Absent / Attendance Rate % / Unknown

### Sidebar Info
- **Database:** Shows enrolled student count
- **Threshold:** Current cosine similarity threshold
- **Embedder:** InceptionResnetV1 / VGGFace2
- **Detector:** MTCNN
- **Enhancer:** GFPGAN status (Enabled / Not installed)

---

## 9. Database Schema

**File:** `attendance.db` (SQLite, auto-created)

```sql
CREATE TABLE IF NOT EXISTS students (
    prn       TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    embedding TEXT NOT NULL
);
```

| Column | Type | Constraint | Description |
|--------|------|------------|-------------|
| `prn` | TEXT | PRIMARY KEY | Unique student roll number |
| `name` | TEXT | NOT NULL | Full student name |
| `embedding` | TEXT | NOT NULL | JSON array of 512 float32 values |

### Notes
- Embeddings are stored as JSON strings and deserialized to `numpy float32` arrays at load time
- `INSERT OR REPLACE` is used — re-enrolling updates the embedding without creating a duplicate
- The database file is fully portable — just copy `attendance.db` to run on any machine

---

## 10. Data Augmentation Strategy

13 augmented variants are generated per face crop during enrollment:

| # | Augmentation | Parameter | Purpose |
|---|-------------|-----------|---------|
| 1 | Original | None | Baseline |
| 2 | Horizontal Flip | Mirror L↔R | Handles left/right face turns |
| 3 | Rotation +15° | Clockwise | Slight head tilt |
| 4 | Rotation -15° | Counter-clockwise | Slight head tilt |
| 5 | Rotation +30° | Clockwise | Moderate tilt |
| 6 | Rotation -30° | Counter-clockwise | Moderate tilt |
| 7 | Rotation +45° | Clockwise | Large tilt |
| 8 | Rotation -45° | Counter-clockwise | Large tilt |
| 9 | Rotation +60° | Clockwise | Near-profile angle |
| 10 | Rotation -60° | Counter-clockwise | Near-profile angle |
| 11 | Brightness Up | +30% | Bright lighting simulation |
| 12 | Brightness Down | -30% | Dim/shadowed lighting |
| 13 | Gaussian Blur | Kernel 3×3 | Low-resolution / motion blur |

### Embedding Math
If a video yields **80 extracted frames**, and MTCNN successfully detects a face in **70 of them**:

```
70 frames × 13 augmentations = 910 embedding vectors
All 910 vectors averaged element-wise
→ 1 master 512-d vector (L2-normalized)
→ Stored in attendance.db
```

This centroid is far more stable and generalizes far better than any single embedding vector.

---

## 11. Configuration Reference

All key parameters can be modified without touching the core logic:

| Parameter | Default | File | Description |
|-----------|---------|------|-------------|
| `COSINE_THRESHOLD` | `0.60` | `model.py` | Min similarity score to mark student Present |
| `FACE_MARGIN` | `20` px | `model.py` | Extra padding added around MTCNN bounding box |
| `MIN_FACE_CONF` | `0.90` | `model.py` | Min MTCNN detection confidence to accept a face |
| `GFPGAN_UPSCALE` | `4` | `model.py` | Upscale factor for GFPGAN restoration |
| `APP_PASSWORD` | `admin123` | `app.py` | Streamlit admin dashboard login password |
| `FRAME_STEP` | `10` | `video_pipeline.py` | Extract 1 frame every N frames from video |
| `MAX_FRAMES` | `80` | `video_pipeline.py` | Max frames to extract per video |
| `EXCEL_NAME_COL` | `Name` | `pipeline.py` | Column header for student name in Excel |
| `EXCEL_PRN_COL` | `PRN No.` | `pipeline.py` | Column header for student PRN in Excel |

---

## 12. How Matching Works

### Cosine Similarity Formula
$$\text{similarity}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}$$

Since all FaceNet embeddings are **L2-normalized** (unit vectors), this simplifies to:
$$\text{similarity}(a, b) = a \cdot b \quad (\text{dot product})$$

### Matching Algorithm
```
For each detected face in the classroom photo:
    live_embedding = FaceNet(face_crop)
    
    best_score = -1.0
    best_student = None
    
    For each enrolled student in database:
        if student already matched: skip  ← no duplicate marking
        score = dot(live_embedding, student.embedding)
        if score > best_score:
            best_score = score
            best_student = student
    
    if best_score >= COSINE_THRESHOLD (0.60):
        mark best_student as PRESENT
    else:
        add face to UNKNOWN list

Students not matched = ABSENT
```

### Score Interpretation
| Score Range | Meaning |
|-------------|---------|
| 0.90 – 1.00 | Near-perfect match (same lighting, pose as enrollment) |
| 0.75 – 0.90 | Strong match (minor variations) |
| 0.60 – 0.75 | Acceptable match (present, accepted) |
| Below 0.60 | Rejected — classified as Unknown |

---

## 13. Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| No face detected during enrollment | Poor video quality or low lighting | Re-record video in good lighting, face clearly visible, no glasses/mask |
| Student marked Unknown in attendance | Score below 60% threshold | Lower `COSINE_THRESHOLD` to `0.50` in `model.py`, or re-enroll with a clearer video |
| `ModuleNotFoundError: facenet_pytorch` | Library not installed | `pip install facenet-pytorch` |
| Excel column not found error | Column headers don't match config | Ensure Excel has exactly `Name` and `PRN No.` headers |
| GFPGAN not available | Not installed | `pip install gfpgan` (optional) |
| Pipeline is very slow (CPU) | No CUDA GPU detected | Install PyTorch with CUDA from pytorch.org |
| `attendance.db` is empty / not found | Enrollment pipeline never run | Run `python pipeline.py` with videos in `Videos/` folder |
| Video not found in Excel | Filename spelling doesn't match Name column | Check spelling: `John_Doe.mp4` must match `John Doe` in Excel |
| Two faces detected from laptop screen | Screen shows another person's face | Ensure no other face is visible on screens in classroom photo |

---

## 14. Resume Summary

### Project Title
**AI-Powered Smart Classroom Attendance System Using Facial Recognition**

### Platform Used
Python Libraries — PyTorch, OpenCV, Streamlit, SQLite, Pandas, NumPy, facenet-pytorch, GFPGAN

### Models Used

| Model | Type | Role |
|-------|------|------|
| MTCNN | CNN Cascade | Multi-face detection & alignment |
| FaceNet (InceptionResnetV1) | Deep CNN / Transfer Learning | 512-d facial embedding generation |
| GFPGAN v1.3 | Generative Adversarial Network | Face restoration & 4× super-resolution |

### Resume Bullet Points

> - Built an **end-to-end AI-based biometric attendance system** using Computer Vision and Deep Learning to automatically detect and recognize multiple students from a single classroom photo, eliminating manual roll calls.
>
> - Implemented **MTCNN** for multi-face detection and **FaceNet (InceptionResnetV1/VGGFace2)** via Transfer Learning to generate 512-d facial embeddings matched using Cosine Similarity.
>
> - Developed a **video-based enrollment pipeline** with OpenCV applying **13× Data Augmentation** (flip, brightness, rotation, blur) and storing embeddings in SQLite.
>
> - Integrated **GFPGAN v1.3 (GAN)** for AI-based face restoration on blurry crops, significantly improving recognition accuracy on wide-angle classroom photographs.
>
> - Deployed a secure **Streamlit** admin dashboard with real-time attendance visualization (Present/Absent/Unknown) and confidence score display.

### Keywords
`Computer Vision` · `Facial Recognition` · `Deep Learning` · `Transfer Learning` · `MTCNN` · `FaceNet` · `GAN` · `GFPGAN` · `Cosine Similarity` · `Data Augmentation` · `Biometric Attendance` · `SQLite` · `Streamlit` · `OpenCV` · `PyTorch`

---

*Documentation generated for: AI-Powered Smart Classroom Attendance System | Version 1.0 | 2026*
