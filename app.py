import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


model = YOLO('best.pt')

st.title("YOLOv8 Human Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    resized_img = cv2.resize(image_np, (640, 640)) 
    results = model(resized_img)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    for r in results:
        im_array = r.plot() 
        im = Image.fromarray(im_array[..., ::-1])  
        st.image(im, caption="Detected Objects", use_column_width=True)