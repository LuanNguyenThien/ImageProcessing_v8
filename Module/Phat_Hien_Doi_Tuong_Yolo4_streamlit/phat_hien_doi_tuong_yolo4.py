import streamlit as st
from PIL import Image
import cv2
import numpy as np

confThreshold = 0.5
nmsThreshold = 0.4
classes = None
with open(r'./Module/Phat_Hien_Doi_Tuong_Yolo4_streamlit/object_detection_classes_yolov4.txt', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def postprocess(frame, outs, outNames):
    frame = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = st.session_state["Net5"].getLayerNames()
    lastLayerId = st.session_state["Net5"].getLayerId(layerNames[-1])
    lastLayer = st.session_state["Net5"].getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # NMS is used inside Region layer only on DNN_BACKEND_OPENcv2 for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    if len(outNames) > 1 or lastLayer.type == 'Region' and 0 != cv2.dnn.DNN_BACKEND_OPENcv2:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box  = boxes[class_indices].tolist()
            nms_indices = cv2.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            nms_indices = nms_indices[:] if len(nms_indices) else []
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    return frame

def nhan_dang_doi_tuong():
    #Mô tả
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        st.header('NHẬN DẠNG ĐỐI TƯỢNG')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                #### Mô tả
                Module này triển khai một ứng dụng nhận dạng đối tượng sử dụng mô hình YOLOv4. Phần nhận diện đối tượng sử dụng DNN module trong thư viện OpenCV chứa các hàm và lớp liên quan đến Deep Neural Networks (DNN), nơi hỗ trợ triển khai và sử dụng các mô hình học sâu (deep learning) trong quá trình xử lý hình ảnh và video. Sau đó dùng module này tiến hành đọc 1 mô hình nhận dạng đối tượng YOLO v4 từ các hình ảnh được truyền vào.
                """
            )
        with col2:
            st.markdown(
                """
                #### Model được sử dụng
                🔻YOLO v4

                🔻yolov4.weights

                🔻object_detection_classes_yolov4.txt

                🔻yolov4.cfg

                """
            )
        st.divider()
        
    try:
        if st.session_state["LoadModel5"] == True:
            print('Đã load model')
            pass
    except:
        st.session_state["LoadModel5"] = True
        st.session_state["Net5"] = cv2.dnn.readNet(r'./Module/Phat_Hien_Doi_Tuong_Yolo4_streamlit/yolov4.weights',
                                                   r'./Module/Phat_Hien_Doi_Tuong_Yolo4_streamlit/yolov4.cfg')
        print('Load model lần đầu')
    st.session_state["Net5"].setPreferableBackend(0)
    st.session_state["Net5"].setPreferableTarget(0)
    outNames = st.session_state["Net5"].getUnconnectedOutLayersNames()

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        image_file = st.file_uploader("Upload Images", type=["bmp", "png", "jpg", "jpeg"])
        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption=None)
            # Chuyển sang cv2 để dùng sau này
            frame = np.array(image)
            frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
            if st.button('Predict'):
                # Process image.
                inpWidth = 416
                inpHeight = 416
                blob = cv2.dnn.blobFromImage(frame.copy(), size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U)
                # Run a model
                st.session_state["Net5"].setInput(blob, scalefactor=0.00392, mean=[0, 0, 0])
                outs = st.session_state["Net5"].forward(outNames)
                img = postprocess(frame, outs, outNames)
                st.image(img, caption=None, channels="BGR")

if __name__ == '__main__':
    nhan_dang_doi_tuong()