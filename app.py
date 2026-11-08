import json
import numpy as np
import streamlit as st
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

st.set_page_config(page_title="CataDetect", page_icon="👁️", layout="centered")

MODEL = "cata_detect.tflite"
META = "cata_detect_meta.json"

meta = json.load(open(META))
THR = float(meta["threshold"])
SIZE = int(meta.get("img_size", 224))

st.markdown("""
<style>
.block-container {max-width: 760px; padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 14px;
       background: linear-gradient(135deg, #0f4c81, #1a7f9c); color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .9;}
.card {padding: 1.1rem 1.3rem; border-radius: 12px; margin-top: .8rem;}
.pos {background: #fdecea; border-left: 6px solid #d93025; color: #5f1410;}
.neg {background: #e6f4ea; border-left: 6px solid #1e8e3e; color: #0d3b1c;}
.card h3 {margin: 0 0 .3rem 0; color: inherit;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>👁️ CataDetect</h1>
  <p>AI-assisted cataract detection from retinal fundus images</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load():
    it = Interpreter(model_path=MODEL)
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]

it, inp, out = load()

def predict(img):
    x = np.expand_dims(np.array(img.resize((SIZE, SIZE)), dtype=np.float32), 0)  # 0-255, no /255
    it.set_tensor(inp["index"], x)
    it.invoke()
    return float(it.get_tensor(out["index"])[0][0])

f = st.file_uploader("Upload a retinal fundus image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if f:
    img = Image.open(f).convert("RGB")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(img, caption="Uploaded image", use_container_width=True)
    with c2:
        with st.spinner("Analysing image..."):
            p = predict(img)
        st.metric("Cataract score", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        st.caption(f"Decision threshold: {THR:.1%}")

    if p >= THR:
        st.markdown("""<div class="card pos"><h3>Possible cataract detected</h3>
        The image shows signs consistent with cataract. Please confirm with an
        ophthalmologist.</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="card neg"><h3>No cataract detected</h3>
        No signs of cataract were found in this image. This does not rule out
        disease, so see an ophthalmologist if you have symptoms.</div>""", unsafe_allow_html=True)

st.divider()
with st.expander("About this tool"):
    st.write(
        "CataDetect uses an EfficientNetB3 deep learning model to classify fundus "
        "images as Normal or Cataract. It supports clinicians and does not replace "
        "an eye examination."
    )
st.caption("AI result for decision support only, not a medical diagnosis.")
