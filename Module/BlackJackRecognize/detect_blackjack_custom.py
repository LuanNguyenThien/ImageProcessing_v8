from ultralytics import YOLO
import cv2
import math
import os


blackjack_classes = [
    '10 chuon', '10 ro', '10 co', '10 bich',
    '2 chuon', '2 ro', '2 co', '2 bich',
    '3 chuon', '3 ro', '3 co', '3 bich',
    '4 chuon', '4 ro', '4 co', '4 bich',
    '5 chuon', '5 ro', '5 co', '5 bich',
    '6 chuon', '6 ro', '6 co', '6 bich',
    '7 chuon', '7 ro', '7 co', '7 bich',
    '8 chuon', '8 ro', '8 co', '8 bich',
    '9 chuon', '9 ro', '9 co', '9 bich',
    'A chuon', 'A ro', 'A co', 'A bich',
    'J chuon', 'J ro', 'J co', 'J bich',
    'Joker',
    'K chuon', 'K ro', 'K co', 'K bich',
    'Q chuon', 'Q ro', 'Q co', 'Q bich',
    'black chip', 'blue chip', 'card back', 'chips', 'green chip', 'red chip', 'white chip'
]


model_blackjack = YOLO("./Module/BlackJackRecognize/weights/best.pt")



def detect_blackjack_video(video_path,result_path):
    os.makedirs(result_path,exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    # Lấy thông số của video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Tạo đối tượng VideoWriter
    out = cv2.VideoWriter(result_path + "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    notice = []
    frame_count = 0
    
    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Gửi frame vào hàm nhận diện
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        str_result, frame = detect_blackjack_frame(frame)

        # Ghi frame đã được xử lý vào video output
        out.write(frame)
        if (frame_count % 10 == 0):
            notice.append("Frame Recognized: "+str(frame_count))
    cap.release()
    out.release()
    for i in range(0,len(notice)):
        notice[i]+="/"+str(frame_count-1)
    notice.append("Frame Recognized: "+str(frame_count-1)+"/"+str(frame_count-1))
    return notice
    
def detect_blackjack_frame(frame):
    result_string = []
    # nhận dạng biển số trước
    plates_detected = model_blackjack(frame,stream=True)
    
    for plate in plates_detected:
        boxes = plate.boxes
        
        for box in boxes:
            # Lấy ra tọa độ của các box biển số
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            blackjack_char = blackjack_classes[int(box.cls[0])]
                   
            
            # In số lên trên biển xe
            org = [x1, y1-1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.2
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, blackjack_char, org, font, fontScale, color, thickness)
            result_string.append("Tìm thấy: " + blackjack_char)
    return result_string,frame
    

                    
