# 🎓 Smart Classroom Attendance System

A robust, AI-powered attendance system built using **MTCNN** (for precise face detection) and **FaceNet** (for facial recognition and embedding generation). 

This project consists of two main parts:
1. **Video Enrollment Pipeline:** Automatically extracts frames from student videos, detects faces, applies data augmentations, and stores the facial embeddings in a SQLite database.
2. **Streamlit Admin Panel:** A sleek web interface where an admin can upload a single classroom photo to instantly mark attendance for all enrolled students.

---

## 🛠️ Prerequisites & Installation

### 1. Requirements
You need Python 3.8 or higher. The system requires the following core libraries:
* `torch` and `torchvision` (PyTorch)
* `facenet-pytorch` (Pre-trained MTCNN and InceptionResnetV1)
* `opencv-python` (cv2 for video/image processing)
* `streamlit` (For the admin web app)
* `pandas` and `openpyxl` (For reading Excel files)
* `Pillow` (PIL for image handling)
* *Optional but recommended:* `gfpgan` (For AI face sharpening on blurry classroom photos)

### 2. Install Dependencies
You can install all required packages by running this command in your terminal:

```bash
pip install torch torchvision facenet-pytorch opencv-python streamlit pandas openpyxl Pillow
```
*(To enable GFPGAN face sharpening, also run: `pip install gfpgan`)*

---

## 🚀 How to Run the Project

### Phase 1: Enroll Students (Database Creation)

Before taking attendance, the system needs to know what the students look like.

1. **Prepare the Data:**
   - Open `students.xlsx`. Add the students' **Name** and **PRN** (Roll Number).
   - Create a folder named `Videos/` in the project directory.
   - Place a short video (e.g., 5-10 seconds) of each student in the `Videos/` folder. Name the video exactly matching their PRN (e.g., `101.mp4` for a student with PRN `101`).

2. **Run the Enrollment Pipeline:**
   Open your terminal and run:
   ```bash
   python pipeline.py
   ```
   *This script will process the videos, extract the best faces, perform data augmentation, generate 512-dimensional embeddings using FaceNet, and save everything into `attendance.db`.*

### Phase 2: Take Attendance (Admin Panel)

Once the database (`attendance.db`) is generated, you can launch the admin dashboard to take attendance from a group photo.

1. **Start the Web App:**
   ```bash
   streamlit run app.py
   ```
2. **Login:**
   - Open the URL provided in the terminal (usually `http://localhost:8501`).
   - Enter the admin password: **`admin123`** (You can change this in `app.py`).
3. **Upload Photo:**
   - Upload a clear photo of the classroom/group of students.
   - The system will detect all faces, match them against the database, and categorize them into **Present**, **Absent**, and **Unknown**.

---

## 📁 Project Structure

* `pipeline.py` — The main orchestrator script for enrolling students from videos.
* `video_pipeline.py` — The worker script that handles video frame extraction, face detection, augmentation, and FaceNet embedding generation.
* `app.py` — The Streamlit admin dashboard for uploading classroom photos and viewing attendance results.
* `model.py` — Centralized AI model loader. Handles MTCNN, FaceNet, and GFPGAN initialization and inference logic.
* `students.xlsx` — Excel sheet containing student Names and PRNs.

---
*Note: Make sure to keep `pipeline.py`, `video_pipeline.py`, `app.py`, and `model.py` in the same directory.*

No change just checking commits
