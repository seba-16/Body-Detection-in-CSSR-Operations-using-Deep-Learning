import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

try:
    model = YOLO('trained_model.pt')
except FileNotFoundError:
    st.error("Error: Model file 'trained_model.pt' not found.")
    st.stop()

st.title("Human Body Detection using Thermal Image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    resized_img = cv2.resize(image_np, (640, 640))
    results = model(resized_img)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        human_count = 0  
        if results and results[0].boxes:
            for r in results:
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])
                im_resized = im.resize(image.size)
                st.image(im_resized, caption="Human Body Detections", use_container_width=True)
                human_count += len(r.boxes)  
            st.success(f"Human Detected | Total humans: {human_count}")
        else:
            st.warning("No Human Detected")