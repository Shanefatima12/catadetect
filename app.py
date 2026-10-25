import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# Use whichever TFLite runtime is installed.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

HERE = Path(__file__).parent
MODEL_PATH = HERE / "cata_detect.tflite"
META_PATH = HERE / "cata_detect_meta.json"

st.set_page_config(page_title="CataDetect", page_icon="👁", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #050d1a 0%, #0a1a2f 100%); color: #e6eef8; }
    .badge { display:inline-block; padding:6px 16px; border:1px solid #1b6f8a; border-radius:999px;
             color:#38bdf8; font-size:11px; letter-spacing:2px; background:#0b2a40; }
    .title { font-size:56px; font-weight:800; color:#7dd3fc; text-align:center; margin:8px 0 0 0; }
    .sub { text-align:center; color:#7f9bb8; margin-bottom:28px; }
    .card { border-radius:14px; padding:20px 22px; margin-top:18px; border:1px solid; }
    .ok   { background:#06241b; border-color:#12603f; }
    .warn { background:#2b1d06; border-color:#8a5a12; }
    .label { font-size:11px; letter-spacing:2px; color:#7f9bb8; }
    .result { font-size:26px; font-weight:700; margin:4px 0 8px 0; }
    .disclaimer { border-left:3px solid #1b6f8a; background:#0b1626; padding:10px 14px;
                  color:#9db3cb; font-size:13px; margin-top:18px; border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    meta = json.loads(META_PATH.read_text())
    interp = Interpreter(model_path=str(MODEL_PATH))
    interp.allocate_tensors()
    return interp, meta


def predict(interp, meta, pil_img):
    size = meta["img_size"]
    img = ImageOps.exif_transpose(pil_img).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)[None, ...]  # 0-255; scaling is inside the model
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], arr)
    interp.invoke()
    return float(interp.get_tensor(out["index"]).ravel()[0])


st.markdown("<div style='text-align:center'><span class='badge'>AI-POWERED MEDICAL IMAGING</span></div>",
            unsafe_allow_html=True)
st.markdown("<div class='title'>CataDetect</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Upload a retinal fundus image to screen for cataract indicators "
            "using deep learning.</div>", unsafe_allow_html=True)

if not (MODEL_PATH.exists() and META_PATH.exists()):
    st.error("Model files not found. Put `cata_detect.tflite` and `cata_detect_meta.json` next to app.py.")
    st.stop()

interp, meta = load_model()

file = st.file_uploader("Drop your fundus eye image here", type=["jpg", "jpeg", "png"],
                        help="Expects a retinal (fundus) photograph, not a normal photo of the face or eye.")

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded fundus image", width=320)

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analysing..."):
            p = predict(interp, meta, image)

        threshold = meta["threshold"]
        suspected = p >= threshold
        css = "warn" if suspected else "ok"
        title = "Cataract suspected" if suspected else "No cataract indicators detected"
        note = ("Signs consistent with cataract were found. Please consult a qualified ophthalmologist "
                "for a proper examination." if suspected else
                "No cataract indicators found in this image. This does not rule out eye disease.")

        st.markdown(
            f"""
            <div class='card {css}'>
              <div class='label'>DIAGNOSTIC RESULT</div>
              <div class='result'>{title}</div>
              <div>{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(p, 0.0), 1.0), text=f"Model cataract score: {p * 100:.1f}%")
        c1, c2 = st.columns(2)
        c1.metric("Cataract score", f"{p * 100:.1f}%")
        c2.metric("Decision threshold", f"{threshold * 100:.0f}%")

st.markdown(
    "<div class='disclaimer'><b>Medical disclaimer:</b> This tool is for educational and screening "
    "purposes only and does not replace a professional medical diagnosis. Please consult a qualified "
    "ophthalmologist for clinical decisions.</div>",
    unsafe_allow_html=True,
)
st.caption("CataDetect - part of VisionExa - built with TFLite and Streamlit")