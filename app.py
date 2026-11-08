import json
import numpy as np
import streamlit as st
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL = "cata_detect.tflite"
META = "cata_detect_meta.json"
NAME = "CataDetect"
POSITIVE = "cataract"

meta = json.load(open(META))
THR = float(meta["threshold"])
SIZE = int(meta.get("img_size", 224))

@st.cache_resource
def load():
    it = Interpreter(model_path=MODEL)
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]

it, inp, out = load()

st.title(f"{NAME} - screening prototype")
st.caption("Fundus image screening only. Not a diagnosis.")

f = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])
if f:
    img = Image.open(f).convert("RGB")
    st.image(img, use_container_width=True)
    x = np.expand_dims(np.array(img.resize((SIZE, SIZE)), dtype=np.float32), 0)  # 0-255, do NOT divide by 255
    it.set_tensor(inp["index"], x)
    it.invoke()
    p = float(it.get_tensor(out["index"])[0][0])
    st.write(f"Score: {p:.3f} (threshold {THR:.3f})")
    if p >= THR:
        st.warning(f"Possible signs of {POSITIVE}. This is a screening result only, so please confirm with an ophthalmologist.")
    else:
        st.success(f"No signs of {POSITIVE} detected. This does not rule out disease.")
