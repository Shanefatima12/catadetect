import json
import numpy as np
import streamlit as st
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

st.set_page_config(page_title="CataDetect | Cataract Detector", page_icon="👁", layout="centered")

MODEL_PATH = "cata_detect.tflite"
META_PATH = "cata_detect_meta.json"
meta = json.load(open(META_PATH))
THR = float(meta["threshold"])
SIZE = int(meta.get("img_size", 224))

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #080d14; color: #e8edf5; }
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d2137 0%, #080d14 60%),
                radial-gradient(ellipse at 80% 100%, #0a1f35 0%, transparent 60%);
    min-height: 100vh;
}
.hero { text-align: center; padding: 3rem 1rem 2rem; }
.hero-badge {
    display: inline-block; background: rgba(0,180,216,0.12); border: 1px solid rgba(0,180,216,0.3);
    color: #00b4d8; font-size: 0.75rem; font-weight: 500; letter-spacing: 0.15em;
    text-transform: uppercase; padding: 0.4rem 1rem; border-radius: 100px; margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 3.4rem; font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #90e0ef 50%, #00b4d8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.8rem;
}
.hero-sub { color: #7a92a8; font-size: 1rem; max-width: 480px; margin: 0 auto 2rem; line-height: 1.6; }
.eye-ring { width: 90px; height: 90px; margin: 0 auto 1.5rem; position: relative;
            display: flex; align-items: center; justify-content: center; }
.eye-ring::before { content: ''; position: absolute; inset: 0; border-radius: 50%;
    border: 2px solid transparent; border-top-color: #00b4d8; border-right-color: #0077b6;
    animation: spin 3s linear infinite; }
.eye-emoji { font-size: 2.4rem; animation: pulse 2s ease-in-out infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.08); } }
.upload-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
               border-radius: 20px; padding: 2rem; margin: 1rem 0; }
[data-testid="stFileUploader"] { background: rgba(0,180,216,0.04) !important;
    border: 1.5px dashed rgba(0,180,216,0.35) !important; border-radius: 16px !important; padding: 1.5rem !important; }
.stButton > button { width: 100%; background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.85rem 2rem !important; font-family: 'Syne', sans-serif !important; font-size: 1rem !important;
    font-weight: 700 !important; margin-top: 1rem !important; box-shadow: 0 4px 24px rgba(0,180,216,0.3) !important; }
.result-card { border-radius: 20px; padding: 2rem; margin-top: 1.5rem; }
.result-grade { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; margin-bottom: 0.4rem; }
.result-desc { color: #a0b4c4; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1.5rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #7a92a8;
              margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; }
.conf-bar-bg { background: rgba(255,255,255,0.06); border-radius: 100px; height: 8px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 100px; }
.stats-row { display: flex; gap: 1rem; margin-top: 1rem; }
.stat-box { flex: 1; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px; padding: 1rem; text-align: center; }
.stat-val { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #00b4d8; }
.stat-lbl { font-size: 0.72rem; color: #7a92a8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }
.footer { text-align: center; color: #3a4f62; font-size: 0.78rem; padding: 2rem 1rem;
          border-top: 1px solid rgba(255,255,255,0.05); margin-top: 3rem; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Medical Imaging</div>
    <div class="eye-ring"><span class="eye-emoji">👁</span></div>
    <div class="hero-title">CataDetect</div>
    <div class="hero-sub">Upload a fundus eye image to instantly detect cataract using deep learning.</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    img = image.convert("RGB").resize((SIZE, SIZE))
    return np.array(img).astype(np.float32)   # keep 0-255: EfficientNet rescales inside the model

st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop your fundus eye image here", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="Uploaded Fundus Image", use_container_width=True)

    if st.button("Run Analysis"):
        with st.spinner("Analyzing retinal patterns..."):
            img_input = np.expand_dims(preprocess_image(image), axis=0)
            interpreter.set_tensor(input_details[0]["index"], img_input)
            interpreter.invoke()
            score = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

        positive = score >= THR
        if positive:
            name, badge, color = "Possible Cataract", "FLAG", "#ef476f"
            desc = "Signs consistent with cataract were detected. Please confirm with an ophthalmologist."
            bg = "background: rgba(239,71,111,0.08); border: 1px solid rgba(239,71,111,0.25);"
            bar = "linear-gradient(90deg, #ef476f, #d62246)"
        else:
            name, badge, color = "No Cataract", "OK", "#06d6a0"
            desc = "No signs of cataract detected in this image. This does not rule out disease, so see an ophthalmologist if you have symptoms."
            bg = "background: rgba(6,214,160,0.08); border: 1px solid rgba(6,214,160,0.25);"
            bar = "linear-gradient(90deg, #06d6a0, #02c39a)"

        st.markdown(f"""
        <div class="result-card" style="{bg}">
            <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.6rem;">
                <span style="font-size:1.2rem; background:{color}; color:#000; padding:0.3rem 0.7rem; border-radius:8px; font-weight:700;">{badge}</span>
                <div>
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:#7a92a8;">Detection Result</div>
                    <div class="result-grade" style="color:{color};">{name}</div>
                </div>
            </div>
            <div class="result-desc">{desc}</div>
            <div class="conf-label">
                <span>Cataract Score</span>
                <span style="color:{color}; font-weight:600;">{score*100:.1f}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{min(score*100,100):.1f}%; background:{bar};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-box"><div class="stat-val">{score*100:.1f}%</div><div class="stat-lbl">Cataract Score</div></div>
            <div class="stat-box"><div class="stat-val">{THR*100:.1f}%</div><div class="stat-lbl">Threshold</div></div>
            <div class="stat-box"><div class="stat-val">{name}</div><div class="stat-lbl">Classification</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1.5rem; padding:1rem; background:rgba(255,255,255,0.03); border-radius:12px; border-left:3px solid #0077b6;">
            <div style="font-size:0.78rem; color:#7a92a8; line-height:1.6;">
                <strong style="color:#a0b4c4;">Medical Disclaimer:</strong>
                This tool is for screening purposes only and does not replace professional medical diagnosis.
                Please consult a qualified ophthalmologist for clinical decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    CataDetect - Built with TFLite and Streamlit - ODIR-5K Dataset<br>
    <span style="color:#1e3448;">For educational and screening purposes only</span>
</div>
""", unsafe_allow_html=True)
