import streamlit as st
import numpy as np
from PIL import Image
import cv2
import shutil

# Xóa trạng thái đã lưu của session
if "LoadModel9" in st.session_state:
        del st.session_state["LoadModel9"]
try:
    if st.session_state["LoadModel9"] == True:
        print('Đã load model rồi')
except:
    st.session_state["LoadModel9"] = True
    st.session_state["Net4"] = cv2.dnn.readNet(
        r".\\Module\\NhanDangTraiCay_Onnx_Streamlit\\yolov8n.onnx")
    print(st.session_state["LoadModel9"])
    print('Load model lần đầu')

filename_classes =  r".\\Module\\NhanDangTraiCay_Onnx_Streamlit\\object_detection_classes_yolo.txt"
mywidth  = 640
myheight = 640
postprocessing = 'yolov8'
background_label_id = -1
backend = 0
target = 0

# Load names of classes
classes = None
if filename_classes:
    with open(filename_classes, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
        print(classes)

st.session_state["Net4"].setPreferableBackend(0)
st.session_state["Net4"].setPreferableTarget(0)
outNames = st.session_state["Net4"].getUnconnectedOutLayersNames()

confThreshold = 0.5
nmsThreshold = 0.4
scale = 0.00392
mean = [0, 0, 0]

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = st.session_state["Net4"].getLayerNames()
    lastLayerId = st.session_state["Net4"].getLayerId(layerNames[-1])
    lastLayer = st.session_state["Net4"].getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'Region' or postprocessing == 'yolov8':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        if postprocessing == 'yolov8':
            box_scale_w = frameWidth / mywidth
            box_scale_h = frameHeight / myheight
        else:
            box_scale_w = frameWidth
            box_scale_h = frameHeight

        for out in outs:
            if postprocessing == 'yolov8':
                out = out[0].transpose(1, 0)

            for detection in out:
                scores = detection[4:]
                if background_label_id >= 0:
                    scores = np.delete(scores, background_label_id)
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * box_scale_w)
                    center_y = int(detection[1] * box_scale_h)
                    width = int(detection[2] * box_scale_w)
                    height = int(detection[3] * box_scale_h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    if len(outNames) > 1 or (lastLayer.type == 'Region' or postprocessing == 'yolov8') and 0 != cv2.dnn.DNN_BACKEND_OPENCV:
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
    return


def nhan_dang_trai_cay():
    #Mô tả
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
                """
                #### Mô tả
                Module nhận dạng 5 loại trái cây triển khai một hệ thống nhận dạng đối tượng dựa 
                trên mô hình ONNX. Em đã tiến hành training lại model này với 5 loại trái cây là 
                Apple, Mango, Orange, Pear, Watermelon, sau đó xuất ra dưới dạng file onnx 
                và sử dụng thư viện OpenCV và ONNX Runtime để thực hiện dự đoán trên ảnh đầu vào và xác định loại trái cây.
                """
        )
    with col2:
        st.markdown(
                """
                #### Model được sử dụng
                🔻nhan_dang_trai_cay.onnx
                """
        )
    st.divider()

    col1, col2, col3 = st.columns([1, 9, 1])
    with col2:
        img_file_buffer = st.file_uploader(
            "Upload an image", type=["bmp", "png", "jpg", "jpeg"])

        if img_file_buffer is not None:
            col4, col5 = st.columns(2)
            image = Image.open(img_file_buffer)
            # Chuyển sang cv2 để dùng sau này
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print(frame)

            col4.image(image, caption='Input', use_column_width=True)
            if st.button('Predict'):
                if not frame is None:
                    frameHeight = frame.shape[0]
                    frameWidth = frame.shape[1]

                    # Create a 4D blob from a frame.
                    inpWidth = mywidth if mywidth else frameWidth
                    inpHeight = myheight if myheight else frameHeight
                    blob = cv2.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U)

                    # Run a model
                    st.session_state["Net4"].setInput(blob, scalefactor=scale, mean=mean)
                    if st.session_state["Net4"].getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                        frame = cv2.resize(frame, (inpWidth, inpHeight))
                        st.session_state["Net4"].setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

                    outs = st.session_state["Net4"].forward(outNames)
                    postprocess(frame, outs)
                    """
                    Put efficiency information. The function getPerfProfile returns       the overall time for inference(t) 
                    and the timings for each of the layers(in layersTimes).
                    """
                    t, _ = st.session_state["Net4"].getPerfProfile()
                    label = 'Inference time: %.2f ms' % (
                        t * 1000.0 / cv2.getTickFrequency())
                    print(label)
                    cv2.putText(frame, label, (20, 30), FONT_FACE,
                                FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
                    img_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    col5.image(img_color, channels='RGB', caption='Output', use_column_width=True)

                if st.button("Xoá bộ nhớ"):
                    st.experimental_rerun()
                    shutil.rmtree(img_file_buffer)
