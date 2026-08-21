# AI-Powered Smart Classroom Attendance System

Built an end-to-end AI-based biometric attendance system using Computer Vision and Deep Learning to automatically detect and recognize multiple students from a single classroom photo, eliminating manual roll calls. 

Implemented **MTCNN** for multi-face detection and **FaceNet (InceptionResnetV1/VGGFace2)** via Transfer Learning to generate 512-d facial embeddings matched using Cosine Similarity. The system features a robust **Streamlit admin dashboard** with a multi-class SQLite database for enrolling students via video (with 13× data augmentation) and generating date-wise attendance reports. Optionally uses **GFPGAN v1.3** to restore and sharpen blurry faces from wide-angle classroom shots.

---

## 🚀 Setup & Installation

### 1. Prerequisites
You need Python 3.8 or higher installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/Attendance_Using_facenet.git
cd Attendance_Using_facenet
```

### 3. Create a Virtual Environment & Install Dependencies
It is highly recommended to use an isolated virtual environment.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> **Note for NVIDIA GPU users:** The `requirements.txt` installs the CPU-only version of PyTorch by default. For faster embedding generation, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Run the Dashboard
```bash
streamlit run app.py
```

* Navigate to `http://localhost:8501` in your browser.
* **Default Admin Password:** `admin123`

### 5. How to Use
1. **Dashboard:** Create a new class (e.g., "CS 101").
2. **Add Student:** Open the class, click "Add Student", and upload a short `.mp4` video (or image) of the student. The system will automatically extract frames and generate their FaceNet embedding.
3. **Take Attendance:** Click "Take Attendance", upload a photo of the entire classroom, and the system will automatically match faces and mark students Present/Absent.
4. **View Reports:** Export date-wise CSV attendance reports directly from the UI.
