# GAN-Assisted Intelligent Classroom Attendance System

## Tech Stack
* **Language:** Python
* **Vision & Detection:** OpenCV, MTCNN
* **Embedding Model:** FaceNet / ArcFace 
* **Enhancement:** ESRGAN (Super-Resolution)
* **Database:** MySQL

## System Architecture & Guardrails
**CRITICAL:** Do NOT train a CNN classifier to identify students. We are using an Embedding Model to generate mathematical vectors. Do NOT use the GAN to generate training data; the GAN is ONLY used to sharpen live, low-resolution CCTV footage.

### Phase 1: One-Time Setup (Master Database)
1. Capture 5 profile images per student (front, up, down, left, right).
2. Detect and crop faces using OpenCV/MTCNN.
3. Pass clean cropped faces through FaceNet to generate 128-dimensional vectors.
4. Save vectors and student IDs to the MySQL database.

### Phase 2: Daily Automation (Live CCTV)
1. Ingest the daily wide-angle CCTV image.
2. Detect and crop all faces in the frame.
3. Pass cropped, low-res faces through ESRGAN to reconstruct and sharpen them.
4. Pass enhanced faces through FaceNet to get live vectors.
5. Calculate Cosine Similarity against the master MySQL database. If the match is >85%, mark the student as present.

s