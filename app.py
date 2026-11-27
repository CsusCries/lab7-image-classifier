import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import efficientnet, mobilenet_v2, resnet50

# Google Drive share link of best_model.keras ("Anyone with the link" can view)
MODEL_URL = "https://drive.google.com/file/d/1V7YEJtvvdo-3kCZxpWgXuhzMcBRryhIb/view?usp=sharing"
MODEL_PATH = "best_model.keras"

PREPROCESS = {
    "MobileNetV2": mobilenet_v2.preprocess_input,
    "ResNet50": resnet50.preprocess_input,
    "EfficientNetB0": efficientnet.preprocess_input,
}


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        import gdown
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    with open("model_info.json") as f:
        info = json.load(f)
    return tf.keras.models.load_model(MODEL_PATH), info


st.title("Cat vs Dog Image Classifier")

with st.spinner("Loading model (first start can take a minute)..."):
    model, info = load_model()

classes = info["class_names"]
size = tuple(info["img_size"])
st.caption(f"Transfer learning model: {info['model_name']} (pre-trained on ImageNet)")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = np.array(img.resize(size), dtype=np.float32)[None, ...]
    x = PREPROCESS[info["model_name"]](x)
    p = float(model.predict(x, verbose=0)[0][0])  # probability of class index 1

    label = classes[1] if p >= 0.5 else classes[0]
    conf = p if p >= 0.5 else 1 - p
    st.subheader(f"Prediction: {label.rstrip('s').capitalize()}")
    st.write(f"Confidence: {conf * 100:.2f}%")
    st.progress(conf)
