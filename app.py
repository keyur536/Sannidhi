"""
app.py  -  Phase 3: MVP Admin Attendance Panel
===============================================
Streamlit web app that:
  1. Authenticates the admin via a password gate.
  2. Accepts a classroom photo upload.
  3. Runs MTCNN to detect & crop every face in the photo.
  4. Enhances each crop with GFPGAN (GAN face restoration).
  5. Embeds each crop with InceptionResnetV1 (vggface2).
  6. Compares against attendance.db via Cosine Similarity (threshold 60%).
  7. Displays Present / Absent / Unknown split with face crops.

Run:
    streamlit run app.py
"""

import io
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Import all models & helpers from model.py ─────────────────────────────────
from model import (
    load_models, load_gfpgan, enhance_with_gfpgan,
    detect_and_crop_faces, match_faces_to_students, load_database,
    COSINE_THRESHOLD, FACE_MARGIN, MIN_FACE_CONF,
    GFPGAN_UPSCALE, GFPGAN_AVAILABLE, DB_PATH,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
FAILED_DIR        = BASE_DIR / "Failed_Detections"
FAILED_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
APP_PASSWORD      = "admin123"

# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit page config  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Attendance Admin Panel",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── App background ── */
  .stApp { background: #0f1117; color: #e8eaf0; }

  /* ── Top header bar ── */
  .top-bar {
    background: linear-gradient(135deg, #1a1d2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 28px;
    border: 1px solid rgba(99,179,237,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .top-bar h1 { color: #63b3ed; margin: 0; font-size: 1.8rem; font-weight: 700; }
  .top-bar p  { color: #a0aec0; margin: 4px 0 0; font-size: 0.9rem; }

  /* ── Stat pill ── */
  .stat-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .stat-pill .val { font-size: 2rem; font-weight: 700; }
  .stat-pill .lbl { font-size: 0.78rem; color: #a0aec0; margin-top: 2px; }

  /* ── Section headers ── */
  .section-present { color: #68d391; font-size: 1.1rem; font-weight: 600; margin-bottom: 12px; }
  .section-absent  { color: #fc8181; font-size: 1.1rem; font-weight: 600; margin-bottom: 12px; }

  /* ── Present card ── */
  .present-card {
    background: linear-gradient(135deg, rgba(104,211,145,0.08) 0%, rgba(72,187,120,0.04) 100%);
    border: 1px solid rgba(104,211,145,0.25);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
  }
  .present-card:hover { border-color: rgba(104,211,145,0.55); }
  .present-card .student-name { font-size: 1rem; font-weight: 600; color: #e8eaf0; }
  .present-card .prn-tag {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    color: #63b3ed;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 4px;
  }
  .present-card .conf-tag {
    display: inline-block;
    background: rgba(104,211,145,0.15);
    color: #68d391;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 4px;
    margin-left: 6px;
  }

  /* ── Absent pill ── */
  .absent-pill {
    background: rgba(252,129,129,0.06);
    border: 1px solid rgba(252,129,129,0.18);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 8px;
    font-size: 0.88rem;
  }
  .absent-pill .absent-name { color: #e8eaf0; font-weight: 500; }
  .absent-pill .absent-prn  { color: #718096; font-size: 0.78rem; }

  /* ── Unknown face card ── */
  .section-unknown { color: #f6ad55; font-size: 1.1rem; font-weight: 600; margin-bottom: 12px; }
  .unknown-card {
    background: rgba(246,173,85,0.06);
    border: 1px solid rgba(246,173,85,0.22);
    border-radius: 12px;
    padding: 10px 12px;
    text-align: center;
    font-size: 0.78rem;
    color: #a0aec0;
  }
  .unknown-card .score-tag {
    display: inline-block;
    background: rgba(246,173,85,0.15);
    color: #f6ad55;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin-top: 6px;
  }

  /* ── Login card ── */
  .login-wrap {
    max-width: 400px;
    margin: 80px auto 0;
    background: linear-gradient(135deg, #1a1d2e, #16213e);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 40px 36px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }
  .login-title { text-align:center; color:#63b3ed; font-size:1.5rem; font-weight:700; margin-bottom:8px; }
  .login-sub   { text-align:center; color:#718096; font-size:0.85rem; margin-bottom:28px; }

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {
    background: rgba(99,179,237,0.04);
    border: 2px dashed rgba(99,179,237,0.3);
    border-radius: 14px;
    padding: 8px;
  }

  /* ── Progress / spinner ── */
  .stSpinner > div { border-top-color: #63b3ed !important; }

  /* ── Divider ── */
  hr { border-color: rgba(255,255,255,0.07) !important; }

  /* ── Buttons ── */
  .stButton > button {
    border-radius: 8px;
    font-size: 0.8rem;
    padding: 4px 14px;
    border: 1px solid rgba(252,129,129,0.4);
    color: #fc8181;
    background: rgba(252,129,129,0.08);
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: rgba(252,129,129,0.2);
    border-color: #fc8181;
  }
  /* Hide Streamlit's deploy button */
  .stDeployButton { display: none; }

  /* Hide default Streamlit footer */
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Cached model loaders  (run once per session)
#  All model logic lives in model.py — these wrappers add Streamlit caching.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _cached_load_models():
    return load_models()


@st.cache_resource(show_spinner=False)
def _cached_load_gfpgan():
    return load_gfpgan()



@st.cache_data(show_spinner=False)
def _cached_load_database():
    return load_database()


# Inference helpers now live in model.py — imported at the top.


def pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  Login gate
# ══════════════════════════════════════════════════════════════════════════════

def render_login() -> None:
    st.markdown("""
    <div class="login-wrap">
      <div class="login-title">🎓 Attendance System</div>
      <div class="login-sub">Admin access only. Enter your password to continue.</div>
    </div>
    """, unsafe_allow_html=True)

    # Centre the form elements under the card
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter admin password",
            label_visibility="collapsed",
        )
        if st.button("Login", use_container_width=True):  # use_container_width kept for button — not deprecated
            if password == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main admin panel
# ══════════════════════════════════════════════════════════════════════════════

def render_panel() -> None:
    # ── Header ──────────────────────────────────────────────────────────────
    now = datetime.now().strftime("%A, %d %B %Y  |  %I:%M %p")
    st.markdown(f"""
    <div class="top-bar">
      <h1>🎓 Classroom Attendance Admin Panel</h1>
      <p>{now}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: logout + DB stats ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Session")
        if st.button("🔒 Logout"):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        db_records = _cached_load_database()
        st.markdown(f"**Database:** `{DB_PATH.name}`")
        st.metric("Enrolled Students", len(db_records))
        st.markdown("---")
        st.markdown(f"**Threshold:** `{int(COSINE_THRESHOLD*100)}%` cosine similarity")
        st.markdown(f"**Embedder:** InceptionResnetV1 / VGGFace2")
        st.markdown(f"**Detector:** MTCNN")
        gfpgan_status = "Enabled" if GFPGAN_AVAILABLE else "Not installed"
        st.markdown(f"**Enhancer:** GFPGAN v1.3 ({gfpgan_status})")
        st.markdown("---")
        use_gfpgan = st.toggle(
            "Enable GFPGAN Enhancement",
            # Activate GFPGAN
            value=False,  
            # value=GFPGAN_AVAILABLE,                                # OFF by default — enable from sidebar when needed
            disabled=not GFPGAN_AVAILABLE,
            help="Sharpens each detected face before embedding. Slightly slower but improves accuracy on blurry images.",
        )

    # ── Load models (cached) ─────────────────────────────────────────────────
    with st.spinner("Loading models (first run only)..."):
        mtcnn, facenet, device = _cached_load_models()
        db_records = _cached_load_database()
        gfpgan_restorer = _cached_load_gfpgan() if GFPGAN_AVAILABLE else None

    if not db_records:
        st.warning(
            f"No student records found in `{DB_PATH}`. "
            "Please run `update_database.py` first."
        )
        return

    # ── Upload section ───────────────────────────────────────────────────────
    st.markdown("### 📷 Upload Classroom Photo")
    uploaded = st.file_uploader(
        label="Drag & drop or browse for a classroom image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("Upload a classroom photo above to begin attendance processing.")
        return

    # Track upload changes
    st.session_state["_last_upload"] = uploaded.name

    # ── Show uploaded image ──────────────────────────────────────────────────
    classroom_img = Image.open(uploaded).convert("RGB")
    with st.expander("📸 View Uploaded Classroom Photo", expanded=False):
        st.image(classroom_img, width="stretch", caption=uploaded.name)

    st.markdown("---")

    # ── Processing pipeline ──────────────────────────────────────────────────
    with st.spinner("Detecting faces with MTCNN..."):
        face_crops = detect_and_crop_faces(classroom_img, mtcnn)

    if not face_crops:
        st.error(
            "MTCNN detected no faces in the uploaded image. "
            "Try a clearer photo with better lighting."
        )
        return

    st.success(f"Detected **{len(face_crops)}** face(s) in the photo.")

    # Replaces the old ESRGAN placeholder.
    # Each detected face crop is passed through GFPGAN to sharpen and
    # restore fine facial details before FaceNet embedding.
    if use_gfpgan and gfpgan_restorer is not None:
        with st.spinner(f"Enhancing {len(face_crops)} face(s) with GFPGAN ({GFPGAN_UPSCALE}x)..."):
            inference_crops = [
                (enhance_with_gfpgan(crop, gfpgan_restorer), bbox)
                for crop, bbox in face_crops
            ]
        st.info(f"GFPGAN enhanced {len(inference_crops)} face crop(s).")
    else:
        inference_crops = face_crops
        if not GFPGAN_AVAILABLE:
            st.caption("GFPGAN not installed — using raw crops. Run: pip install gfpgan")

    with st.spinner("Generating embeddings & matching against database..."):
        present, absent, unknown = match_faces_to_students(
            inference_crops, db_records, facenet, device, COSINE_THRESHOLD
        )

    # ── Stats bar ────────────────────────────────────────────────────────────
    total     = len(db_records)
    n_present = len(present)
    n_absent  = len(absent)
    n_unknown = len(unknown)
    pct       = int(n_present / total * 100) if total > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="stat-pill">
          <div class="val" style="color:#63b3ed">{total}</div>
          <div class="lbl">Enrolled Students</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-pill">
          <div class="val" style="color:#68d391">{n_present}</div>
          <div class="lbl">Present</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-pill">
          <div class="val" style="color:#fc8181">{n_absent}</div>
          <div class="lbl">Absent</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-pill">
          <div class="val" style="color:#f6ad55">{pct}%</div>
          <div class="lbl">Attendance Rate</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="stat-pill">
          <div class="val" style="color:#b794f4">{n_unknown}</div>
          <div class="lbl">Unknown Faces</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results columns ──────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    # ════════ PRESENT ════════
    with left_col:
        st.markdown(
            f'<div class="section-present">✅ Present Students ({n_present})</div>',
            unsafe_allow_html=True,
        )

        if not present:
            st.markdown(
                '<div class="absent-pill"><span class="absent-name">No students identified as present.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            # Sort by name for clean display
            present_sorted = sorted(present, key=lambda x: x["name"])

            for match in present_sorted:
                conf_pct = int(match["score"] * 100)

                img_col, info_col = st.columns([1, 3], gap="small")

                with img_col:
                    st.image(
                        match["crop_img"],
                        width="stretch",
                        caption="",
                    )

                with info_col:
                    st.markdown(f"""
                    <div class="present-card">
                      <div class="student-name">{match['name']}</div>
                      <span class="prn-tag">PRN: {match['prn']}</span>
                      <span class="conf-tag">Confidence: {conf_pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<hr style='margin:4px 0 12px'>", unsafe_allow_html=True)

    # ════════ ABSENT ════════
    with right_col:
        st.markdown(
            f'<div class="section-absent">❌ Absent Students ({n_absent})</div>',
            unsafe_allow_html=True,
        )

        if not absent:
            st.success("All enrolled students are present! 🎉")
        else:
            absent_sorted = sorted(absent, key=lambda x: x["name"])
            for s in absent_sorted:
                st.markdown(f"""
                <div class="absent-pill">
                  <span class="absent-name">{s['name']}</span><br>
                  <span class="absent-prn">PRN: {s['prn']}</span>
                </div>
                """, unsafe_allow_html=True)

    # ════════ UNKNOWN FACES ════════
    st.markdown("---")
    st.markdown(
        f'<div class="section-unknown">&#x26A0; Unknown Faces Detected ({n_unknown})</div>',
        unsafe_allow_html=True,
    )

    if n_unknown == 0:
        st.info("No unknown faces detected — every face in the photo matched a student.")
    else:
        st.caption(
            f"These {n_unknown} face(s) were detected by MTCNN but scored below the "
            f"{int(COSINE_THRESHOLD*100)}% cosine threshold for every enrolled student. "
            "They may be visitors, staff, or students not yet enrolled in the database."
        )
        # Display as a responsive grid (up to 6 per row)
        cols_per_row = min(n_unknown, 6)
        rows = [unknown[i:i+cols_per_row]
                for i in range(0, n_unknown, cols_per_row)]
        for row in rows:
            grid = st.columns(cols_per_row)
            for col, face in zip(grid, row):
                with col:
                    st.image(face["crop_img"], width="stretch", caption="")
                    score_pct = int(face["best_score"] * 100)
                    st.markdown(
                        f'<div class="unknown-card">'
                        f'Unknown<br>'
                        f'<span class="score-tag">Best match: {score_pct}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        render_login()
    else:
        render_panel()


if __name__ == "__main__":
    main()
