from ultralytics import YOLO
import cv2
import math
import streamlit as st
import os
import shutil
from PIL import Image
import numpy as np

model_helmet = YOLO("./Module/HelmetDetection/weights/best.pt")


def detect_helmet_frame(frame):
    result_string = []
    # nhận dạng biển số trước
    helmets_detected = model_helmet(frame, stream=True)

    for helmet in helmets_detected:
        boxes = helmet.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(
                x2), int(y2)  # convert to int values

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    return frame


def runDetectHelmet():
    #Mô tả
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
                #### Mô tả
                Model này được training trên tập dữ liệu custom bằng thư viện OpenCV để phân tích ảnh và scikit-learn. Cho phép phát hiện ra mũ bảo hiểm trong các tấm ảnh được truyền vào.
                """
        )
    with col2:
        st.markdown(
            """
                #### Model được sử dụng
                🔻Tự train trên data custom
                """
        )
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        image_path = st.file_uploader(
            "Upload Images", type=["bmp", "png", "jpg", "jpeg"])
        detect = st.button("Nhận diện", type='primary')
        delete = st.button("Xoá bộ nhớ")
    if image_path is not None:
        image = Image.open(image_path)
        frame = np.array(image)
        col2.image(image, caption='Input')
        if detect:
            frame_output = detect_helmet_frame(frame)
            col2.image(frame_output, caption='Output')
            if delete:
                st.experimental_rerun()
                shutil.rmtree(image_path)
