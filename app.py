import json
import os
import urllib.request

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import efficientnet, mobilenet_v2, resnet50

# Google Drive file ID of best_model.keras (shared as "Anyone with the link")
DRIVE_FILE_ID = "1V7YEJtvvdo-3kCZxpWgXuhzMcBRryhIb"
MODEL_PATH = "best_model.keras"

PREPROCESS = {
    "MobileNetV2": mobilenet_v2.preprocess_input,
    "ResNet50": resnet50.preprocess_input,
    "EfficientNetB0": efficientnet.preprocess_input,
}


def download_model():
    """Download best_model.keras from Google Drive."""
    url = ("https://drive.usercontent.google.com/download"
           f"?id={DRIVE_FILE_ID}&export=download&confirm=t")
    tmp = MODEL_PATH + ".part"
    urllib.request.urlretrieve(url, tmp)
    with open(tmp, "rb") as f:
        if f.read(1) == b"<":  # got an HTML page instead of the model file
            os.remove(tmp)
            raise RuntimeError("Could not download the model. Check that the "
                               "Google Drive file is shared as 'Anyone with the link'.")
    os.replace(tmp, MODEL_PATH)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        download_model()
    with open("model_info.json") as f:
        info = json.load(f)
    return tf.keras.models.load_model(MODEL_PATH), info


st.title("Cat vs Dog Image Classifier")

with st.spinner("Loading model (first start can take a minute)..."):
    model, info = load_model()

classes = info["class_names"]
size = tuple(info["img_size"])
st.caption(f"Transfer learning model: {info['model_name']} (pre-trained on ImageNet)")

# No `type` filter: on Android this lets the picker offer the Camera option
file = st.file_uploader("Upload an image (or take a photo)")
if file:
    try:
        img = Image.open(file).convert("RGB")
    except Exception:
        st.error("That file isn't an image. Please upload a JPG, PNG or WEBP photo.")
        st.stop()
    st.image(img, caption="Input image", use_container_width=True)

    x = np.array(img.resize(size), dtype=np.float32)[None, ...]
    x = PREPROCESS[info["model_name"]](x)
    p = float(model.predict(x, verbose=0)[0][0])  # probability of class index 1

    label = classes[1] if p >= 0.5 else classes[0]
    conf = p if p >= 0.5 else 1 - p
    st.subheader(f"Prediction: {label.rstrip('s').capitalize()}")
    st.write(f"Confidence: {conf * 100:.2f}%")
    st.progress(conf)
