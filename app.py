import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import io
import base64
import time
import os
from pathlib import Path

from model import (
    load_models, load_gfpgan, enhance_with_gfpgan,
    detect_and_crop_faces, match_faces_to_students,
    COSINE_THRESHOLD, MIN_FACE_CONF, GFPGAN_AVAILABLE
)
import db_utils
from video_pipeline import extract_frames, crop_face_from_frame, generate_master_embedding

APP_PASSWORD = "admin123"

st.set_page_config(page_title="AI Smart Attendance", page_icon="🎓", layout="wide")

# CSS
st.markdown("""
<style>
    .card { background-color: #1e253c; border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px; }
    .card:hover { border-color: rgba(99,179,237,0.4); transition: 0.3s; }
    .title-text { color: #63b3ed; font-size: 1.2rem; font-weight: bold; }
    .sub-text { color: #a0aec0; font-size: 0.9rem; }
    .stat-pill { background: rgba(255,255,255,0.05); padding: 8px 15px; border-radius: 20px; margin-right: 10px; font-weight: 600; display: inline-block; }
    .stat-pill-green { color: #68d391; background: rgba(104,211,145,0.1); }
    .stat-pill-red { color: #fc8181; background: rgba(252,129,129,0.1); }
    
    .present-card { background: rgba(104,211,145,0.08); border: 1px solid rgba(104,211,145,0.25); border-radius: 12px; padding: 12px; display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
    .present-card img { border-radius: 8px; width: 45px; height: 45px; object-fit: cover; border: 1px solid rgba(104,211,145,0.5); }
    .present-card .name { color: #e8eaf0; font-weight: 600; }
    .present-card .tags span { background: rgba(99,179,237,0.15); color: #63b3ed; border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; margin-right: 5px;}
    .present-card .tags .conf { background: rgba(104,211,145,0.15); color: #68d391; }
    
    .absent-pill { background: rgba(252,129,129,0.06); border: 1px solid rgba(252,129,129,0.18); border-radius: 10px; padding: 10px 16px; margin-bottom: 8px; font-size: 0.88rem; }
    .absent-pill .name { color: #e8eaf0; font-weight: 500; }
    .absent-pill .prn { color: #718096; font-size: 0.78rem; margin-left: 10px; }
    
    .unknown-card { background: rgba(246,173,85,0.06); border: 1px solid rgba(246,173,85,0.22); border-radius: 12px; padding: 10px; text-align: center; }
    .unknown-card img { border-radius: 8px; width: 45px; height: 45px; object-fit: cover; }
    .unknown-card .score { background: rgba(246,173,85,0.15); color: #f6ad55; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; margin-top: 5px; display: inline-block;}
</style>
""", unsafe_allow_html=True)

# ── Session State ──
if "auth" not in st.session_state:
    st.session_state.auth = False
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "current_class" not in st.session_state:
    st.session_state.current_class = None
if "mtcnn" not in st.session_state:
    st.session_state.mtcnn = None
if "facenet" not in st.session_state:
    st.session_state.facenet = None
if "gfpgan" not in st.session_state:
    st.session_state.gfpgan = None

def load_ai():
    if st.session_state.mtcnn is None:
        with st.spinner("Loading MTCNN & FaceNet..."):
            mtcnn, facenet, device = load_models()
            st.session_state.mtcnn = mtcnn
            st.session_state.facenet = facenet
            st.session_state.device = device
            if GFPGAN_AVAILABLE:
                st.session_state.gfpgan = load_gfpgan()

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def nav_to(page, class_id=None):
    st.session_state.page = page
    if class_id is not None:
        st.session_state.current_class = class_id
    st.rerun()

# ── Login ──
if not st.session_state.auth:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.markdown("<div class='card' style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<h2>🎓 Smart Attendance</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#a0aec0;'>Admin Login</p>", unsafe_allow_html=True)
        pwd = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if pwd == APP_PASSWORD:
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Incorrect Password")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown("### 🎓 Attendance System")
    if st.button("🏠 Dashboard", use_container_width=True):
        nav_to("dashboard")
    
    if st.session_state.current_class:
        cls = db_utils.get_class(st.session_state.current_class)
        st.markdown(f"---")
        st.markdown(f"**Current Class:**\n{cls['class_name']}")
        if st.button("✅ Take Attendance", use_container_width=True):
            nav_to("attendance")
        if st.button("➕ Add Student", use_container_width=True):
            nav_to("enroll")
        if st.button("👥 View Students", use_container_width=True):
            nav_to("view_students")
        if st.button("📊 View Reports", use_container_width=True):
            nav_to("reports")
            
    st.markdown("---")
    st.markdown(f"⚙️ **System Info**")
    st.markdown(f"- Threshold: `{COSINE_THRESHOLD}`")
    st.markdown(f"- GFPGAN: `{'✅' if GFPGAN_AVAILABLE else '❌'}`")
    if st.button("Logout"):
        st.session_state.auth = False
        st.rerun()

# ── Routing ──
page = st.session_state.page

if page == "dashboard":
    st.title("Class Dashboard")
    
    # Create Class Expander
    with st.expander("➕ Create New Class"):
        with st.form("new_class_form"):
            c_name = st.text_input("Class Name*", placeholder="e.g. CS 101 Batch A")
            c_subj = st.text_input("Subject", placeholder="e.g. Data Structures")
            c_desc = st.text_area("Description")
            if st.form_submit_button("Create Class"):
                if c_name.strip():
                    db_utils.create_class(c_name.strip(), c_subj.strip(), c_desc.strip())
                    st.success("Class created!")
                    st.rerun()
                else:
                    st.error("Class Name is required.")

    st.markdown("### Your Classes")
    classes = db_utils.get_all_classes()
    if not classes:
        st.info("No classes found. Create one above.")
    else:
        cols = st.columns(3)
        for i, cls in enumerate(classes):
            c = cols[i % 3]
            with c:
                st.markdown(f"""
                <div class="card">
                    <div class="title-text">{cls['class_name']}</div>
                    <div class="sub-text">{cls['subject']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Manage", key=f"manage_{cls['class_id']}", use_container_width=True):
                    nav_to("attendance", cls['class_id'])

elif page == "enroll":
    cls = db_utils.get_class(st.session_state.current_class)
    st.title(f"Add Student: {cls['class_name']}")
    
    with st.form("enroll_form"):
        s_name = st.text_input("Student Name*")
        s_prn = st.text_input("PRN / Roll No.*")
        uploaded_file = st.file_uploader("Upload Student Video (.mp4) or Image", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])
        
        if st.form_submit_button("Enroll Student"):
            if not s_name or not s_prn or not uploaded_file:
                st.error("All fields and file are required.")
            else:
                load_ai()
                with st.spinner("Processing..."):
                    # Process Video or Image
                    crops = []
                    if uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        
                        try:
                            # Use existing video pipeline logic but imported
                            frames = extract_frames(Path(tmp_path))
                            for frame in frames:
                                crop = crop_face_from_frame(frame, st.session_state.mtcnn)
                                if crop:
                                    crops.append(crop)
                        finally:
                            os.remove(tmp_path)
                    else:
                        # Image processing
                        pil_img = Image.open(uploaded_file).convert('RGB')
                        # Run standard MTCNN detect_and_crop
                        faces = detect_and_crop_faces(pil_img, st.session_state.mtcnn)
                        if faces:
                            # take the largest face (first one if sorted, but detect returns as is, assume first)
                            # Actually detect_and_crop_faces for keep_all=True might return multiple.
                            # Just take the first one for simplicity.
                            crops.append(faces[0][0])
                    
                    if not crops:
                        st.error("No face detected! Please upload a clearer video or image.")
                    else:
                        master_emb = generate_master_embedding(crops, st.session_state.facenet, st.session_state.device)
                        if master_emb is not None:
                            try:
                                db_utils.add_student(cls['class_id'], s_prn, s_name, master_emb)
                                st.success(f"Successfully enrolled {s_name} ({s_prn})!")
                            except Exception as e:
                                st.error(f"Database error: {e}")
                        else:
                            st.error("Failed to generate embedding.")

elif page == "attendance":
    cls = db_utils.get_class(st.session_state.current_class)
    st.title(f"Take Attendance: {cls['class_name']}")
    
    date = st.date_input("Attendance Date", datetime.date.today())
    uploaded_photo = st.file_uploader("Upload Classroom Photo", type=['jpg', 'jpeg', 'png'])
    use_gfpgan = st.checkbox("Enable GFPGAN Face Enhancement", value=False) if GFPGAN_AVAILABLE else False
    
    if uploaded_photo and st.button("Process Attendance", type="primary"):
        load_ai()
        students_in_class = db_utils.get_students_for_class(cls['class_id'])
        if not students_in_class:
            st.warning("No students enrolled in this class yet! Please add students first.")
        else:
            with st.spinner("Detecting and matching faces..."):
                pil_img = Image.open(uploaded_photo).convert("RGB")
                faces = detect_and_crop_faces(pil_img, st.session_state.mtcnn)
                
                if not faces:
                    st.error("No faces found in the image.")
                else:
                    # Optional GFPGAN
                    if use_gfpgan and st.session_state.gfpgan:
                        st.toast("Enhancing faces with GFPGAN...")
                        enhanced_faces = []
                        for crop, box in faces:
                            enh = enhance_with_gfpgan(crop, st.session_state.gfpgan)
                            enhanced_faces.append((enh, box))
                        faces = enhanced_faces
                    
                    present, absent, unknown = match_faces_to_students(
                        face_crops=faces,
                        db_records=students_in_class,
                        facenet=st.session_state.facenet,
                        device=st.session_state.device
                    )
                    
                    # Save to DB
                    db_utils.save_attendance_session(
                        class_id=cls['class_id'],
                        date=date,
                        present_list=present,
                        absent_list=absent,
                        unknown_count=len(unknown)
                    )
                    
                    st.success(f"Attendance saved for {date}!")
                    
                    # ── Display Stats ──
                    st.markdown("---")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Enrolled", len(students_in_class))
                    col2.metric("Present", len(present))
                    col3.metric("Absent", len(absent))
                    rate = (len(present) / len(students_in_class)) * 100 if students_in_class else 0
                    col4.metric("Attendance %", f"{rate:.1f}%")
                    col5.metric("Unknown Faces", len(unknown))
                    
                    # ── Display Details ──
                    st.markdown("### ✅ Present")
                    if present:
                        pc1, pc2, pc3 = st.columns(3)
                        for idx, p in enumerate(present):
                            col = [pc1, pc2, pc3][idx % 3]
                            b64 = pil_to_base64(p["crop_img"])
                            with col:
                                st.markdown(f"""
                                <div class="present-card">
                                    <img src="data:image/jpeg;base64,{b64}">
                                    <div>
                                        <div class="name">{p['name']}</div>
                                        <div class="tags">
                                            <span>{p['prn']}</span>
                                            <span class="conf">{p['score']*100:.1f}% Match</span>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No present students matched.")
                        
                    st.markdown("### ❌ Absent")
                    if absent:
                        for a in absent:
                            st.markdown(f"""
                            <div class="absent-pill">
                                <span class="name">{a['name']}</span>
                                <span class="prn">PRN: {a['prn']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("100% Attendance!")
                        
                    if unknown:
                        st.markdown(f"### ⚠️ Unknown ({len(unknown)})")
                        ucols = st.columns(6)
                        for idx, u in enumerate(unknown):
                            b64 = pil_to_base64(u["crop_img"])
                            with ucols[idx % 6]:
                                st.markdown(f"""
                                <div class="unknown-card">
                                    <img src="data:image/jpeg;base64,{b64}">
                                    <div class="score">{u['best_score']*100:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)

elif page == "view_students":
    cls = db_utils.get_class(st.session_state.current_class)
    st.title(f"Enrolled Students: {cls['class_name']}")
    
    students_in_class = db_utils.get_students_for_class(cls['class_id'])
    
    if not students_in_class:
        st.info("No students are enrolled in this class yet.")
    else:
        st.write(f"**Total Students:** {len(students_in_class)}")
        
        # Format data for display
        display_data = []
        for s in students_in_class:
            display_data.append({
                "PRN / Roll No.": s['prn'],
                "Name": s['name'],
                "Enrolled On": s.get('enrolled_at', 'N/A')
            })
            
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True)

elif page == "reports":
    cls = db_utils.get_class(st.session_state.current_class)
    st.title(f"Reports: {cls['class_name']}")
    
    dates = db_utils.get_session_dates(cls['class_id'])
    if not dates:
        st.info("No attendance taken yet.")
    else:
        sel_date = st.selectbox("Select Date", dates)
        
        report_data = db_utils.get_attendance_report(cls['class_id'], sel_date)
        if report_data:
            df = pd.DataFrame(report_data)
            
            # Format dataframe
            df['Status'] = df['status'].map({'P': 'Present', 'A': 'Absent', 'U': 'Unknown'})
            df['Match Confidence'] = df['confidence_score'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "-")
            df = df[['prn', 'name', 'Status', 'Match Confidence']]
            df.columns = ['PRN', 'Name', 'Status', 'Match Confidence']
            
            st.dataframe(df, use_container_width=True)
            
            # Download buttons
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"Attendance_{cls['class_name']}_{sel_date}.csv",
                mime="text/csv",
            )
