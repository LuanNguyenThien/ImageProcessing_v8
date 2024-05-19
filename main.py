import streamlit as st
import cv2
import time
import shutil
import os
from PIL import Image
import numpy as np

#module
from Module.Hand_volumeAdjust import Nhan_Dang_Tay_Tang_Giam_Am_Luong
from Module.face_recognize import mainFace
from Module.nhan_dang_chu_so_mnist_streamlit.home import runDetect
from Module.NhanDangTraiCay_Onnx_Streamlit import nhan_dang_trai_cay
from Module.faceAgeGender_dectected import mainAgeGender
from Module.HandWriting.handwriting_streamlit import handwriting_streamlit_show 
from Module.BlackJackRecognize.detect_blackjack_custom import detect_blackjack_frame, detect_blackjack_video
from Module.XuLyAnh.xulyanh import runXuLyAnh
from Module.ExamCheatingDetection.ExamCheatingDetect_streamlit import ExamCheatingDetect

from moviepy.editor import VideoFileClip
from ffmpy import FFmpeg

#conver video để xuất được lên streamlit
def convert_video(input_path, output_path):
    codec='libx264' 
    audio_codec='aac'
    try:
        video_clip = VideoFileClip(input_path)
        video_clip.write_videofile(output_path, codec=codec, audio_codec=audio_codec)
        return True
    except Exception as e:
        print(f"Lỗi khi chuyển đổi video: {e}")
        return False

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Đồ án cuối kỳ Xử lý ảnh số",
    layout="wide",
    page_icon="📸",
    initial_sidebar_state="expanded",
)



#Trang home
def display_home():
    #Định dạng chữ
    st.markdown(
    """
    <style>
        .stHeadingContainer span {
            color: #000000;
            font-size: 36px;
            text-align: center;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 20px;
        }

        [data-testid="stMarkdownContainer"] {
            text-align: center;
        }

        img {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True
    )
    
    #hiệu ứng
    
    st.snow()
        
    st.image('./img/HCMUTE-fit.png')
    st.divider()
    st.title("Đồ án cuối kỳ môn Xử lý ảnh số")
    st.write("**Mã lớp: DIPR430685_23_1_02**")
    st.write("**GVHD: ThS. Trần Tiến Đức**")
    st.divider()
    st.subheader("Sinh viên thực hiện")
    col1, col2, col3, col4, col5 = st.columns([1,3,1,3,1])

    with col2:
        st.image("./img/luan.png")
        st.write("**Nguyễn Thiện Luân**")
        st.write("**MSSV: 21110538**")

    with col4:
        st.image("./img/bao.png")
        st.write("**Lê Nguyễn Bảo**")
        st.write("**MSSV: 21110374**")
    
        
    st.divider()    
    st.subheader("Nội dung project")
    col1, col2=st.columns(2)
    with col1:
        st.subheader("4 chức năng chính")
        st.write("😃 Nhận dạng khuôn mặt")
        st.write("🔢 Nhận dạng chữ số viết tay")
        st.write("✈️ Nhận dạng đối tượng")
        st.write("🖼️ Xử lý ảnh")

    with col2:
        st.subheader("Chức năng thêm")
        st.write("👶 Nhận dạng tuổi - giới tính")
        st.write("✍️ Nhận dạng chữ viết tay")
        st.write("🃏 Nhận dạng lá bài tây")
        st.write("🚨 Phát hiện gian lận")
        st.write("✌️ Nhận diện tay điều chỉnh âm lượng")
    
#Nhận dạng khuôn mặt
def display_face_detection():    
    st.title("😃NHẬN DẠNG KHUÔN MẶT")
    st.divider()
    
    mainFace.mainface()
    
#Nhận dạng chữ số viết tay
def display_handwritten_digit_recognition():        
    st.title("🔢NHẬN DẠNG CHỮ SỐ VIẾT TAY")
    st.divider()
    
    runDetect()
    
#Nhận dạng đối tượng
def display_object_detection():    
    st.title("NHẬN DẠNG CÁC LOẠI ĐỐI TƯỢNG")
    st.subheader("Các loại đối tượng đã training model:")
    st.write("person 🚶, bicycle 🚲, car 🚗, motorbike 🏍️, aeroplane ✈️, bus 🚌, train 🚆, truck 🚚, boat ⛵, traffic light 🚦, stop sign 🛑, cat 🐱, dog 🐶, handbag 👜, tie 👔, suitcase 🧳, frisbee 🥏, skis 🎿, snowboard 🏂, sports ball ⚽, kite 🪁, baseball bat ⚾, baseball glove ⚾, skateboard 🛹, surfboard 🏄‍♂️, tennis racket 🎾, bottle 🍾, wine glass 🍷, cup ☕, fork 🍴, knife 🔪, spoon 🥄, bowl 🥣, banana 🍌, apple 🍎, sandwich 🥪, orange 🍊, broccoli 🥦, carrot 🥕, hot dog 🌭, pizza 🍕, donut 🍩, cake 🍰, chair 🪑, sofa 🛋️, bed 🛏️, diningtable 🪑, toilet 🚽, tvmonitor 📺, laptop 💻, mouse 🖱️, remote 📱, keyboard ⌨️, cell phone 📱, microwave 🍲, oven 🍲, toaster 🍞, sink 🚰, refrigerator 🧊, book 📚, clock 🕰️, vase 🏺, scissors ✂️, teddy bear 🧸, hair drier 💇 ...")   
    st.divider()
    
    nhan_dang_trai_cay.nhan_dang_trai_cay() 

#Xử lý ảnh
def display_image_processing():
    st.title("🖼️XỬ LÝ ẢNH")
    st.divider()
    
    runXuLyAnh()

#Nhận dạng tuổi, giới tính
def display_face_Age_Gender():    
    st.title("👶NHẬN DẠNG TUỔI - GIỚI TÍNH")
    st.divider()
    mainAgeGender.runAgeGender()
     
#Nhận dạng chữ viết tay
def hand_writting():
    st.title("✍️NHẬN DẠNG CHỮ VIẾT TAY")
    st.divider()
    handwriting_streamlit_show()
        
#Nhận dạng lá bài tây
def cards():   
    st.title("🃏NHẬN DẠNG LÁ BÀI TÂY")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
                #### Mô tả
                Model này được training trên tập dữ liệu custom về 53 lá bài tây bằng thư viện OpenCV để phân tích ảnh và scikit-learn. Cho phép phát hiện ra các loại bài khác nhau khi truyền ảnh vào
                """
        )
    with col2:
        _col1, _col2 = st.columns(2)
        with _col2:
            st.markdown(
                """
                    #### Model được sử dụng
                    🔻Tự train trên data custom
                    """
            )
    st.divider()
    
    result_path = ".\\Module\\BlackJackRecognize\\result\\"
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, MP4 file",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Scanned file are not supported yet!",
    )
    
    col1, col2 = st.columns(2)
    bt1 = col1.button("Nhận dạng",type="primary")
    bt2 = col2.button("Xóa",type="primary")
    
    if bt1:
        if not uploaded_file:
            st.warning("Vui lòng upload file để nhận dạng!!!")
        else:
            with st.spinner('Wait for it...'):    
                time.sleep(1)
                st.progress(100)
            if "image" in uploaded_file.type:
                st.image(uploaded_file)
                with st.status("Black Jack Recognization Start!", expanded=True) as status:
                    st.write("Open image: "+uploaded_file.name)
                    image = Image.open(uploaded_file)
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    st.write("Recognizing...")
                    result_string, image = detect_blackjack_frame(frame)
                    for s in result_string:
                        st.write(s)
                        time.sleep(0.5)
                    st.write("Done!")
                    status.update(label= "Completed recognize BlackJack",state="complete", expanded=False)
                st.image(image,"Ảnh đã nhận diện (Để biết thêm chi tiết, click vào status)",channels="BGR")
                
            if "video" in uploaded_file.type:
                os.makedirs(result_path,exist_ok=True)
                st.video(uploaded_file) 
                with st.status("Licesen Plate Recognization Start!", expanded=True) as status:
                    st.write("Open video: "+uploaded_file.name)
                    if os.path.exists(result_path):
                        shutil.rmtree(result_path)
                    # lưu video
                    video_bytes = uploaded_file.read()
                    input_filepath = ".\Module\BlackJackRecognize\input.mp4"
                    with open(input_filepath, 'wb') as file:
                        file.write(video_bytes)

                    st.write("Recognizing...")
                    detect_blackjack_video(input_filepath,result_path)
                    st.write("Recognization done!")
                    st.write("Start convert video!")
                    convert_video(input_path=result_path + "output.mp4",
                              output_path=result_path + 'output_convert.mp4')
                    st.write("All done!")
                    status.update(label= "Recognization successfully! Video is available!",state="complete", expanded=False)
                st.video(result_path + 'output_convert.mp4')
    
    if bt2:
        if os.path.exists(".\Module\BlackJackRecognize\input.mp4"):
            os.remove(".\Module\BlackJackRecognize\input.mp4")
        if os.path.exists(result_path):
            shutil.rmtree(result_path)  

#Phát hiện gian lận      
def exam_cheating():
    st.title("🚨Phát hiện gian lận")  
    st.divider()
    
    ExamCheatingDetect()

def hand_volumeAdjust():
    st.title("✌️NHẬN DIỆN TAY ĐIỀU CHỈNH ÂM LƯỢNG")
    st.divider()
    
    Nhan_Dang_Tay_Tang_Giam_Am_Luong.volumeAdjust()
          
#Main    
def main():
    #chèn background
    page_bg_img = """   
    <style>
    # .stApp {
    # background: url("https://cellphones.com.vn/sforum/wp-content/uploads/2023/02/hinh-nen-may-tinh-4k-76.jpg")
    #     no-repeat center center fixed !important;
    # background-size: cover !important;
    # }
    # .stApp::before {
    # content: "";
    # position: absolute;
    # top: 0;
    # left: 0;
    # width: 100%;
    # height: 100%;
    # background-color: rgba(255, 255, 255, 0.5);
    # }
    # [data-testid="stHeader"]{
    #     background: rgba(0,0,0,0);
    # }
    # [data-testid="stToolbar"]{
    #     right:2rem;
    # }
    [data-testid="stSidebar"] > div:first-child {
        background-image: url("https://images.pexels.com/photos/2437299/pexels-photo-2437299.jpeg?auto=compress&cs=tinysrgb&w=800");
        background-position: center;      
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    # Thanh menu
    menu = ["Trang chủ", "Nhận dạng khuôn mặt", "Nhận dạng đối tượng",
            "Nhận dạng chữ số viết tay", "Xử lý ảnh", "Nhận dạng tuổi - giới tính", 
            "Nhận dạng chữ viết tay", "Nhận dạng lá bài tây", "Phát hiện gian lận", "Nhận diện tay điều chỉnh âm lượng"]
    # Biểu tượng tương ứng với mỗi mục trong menu
    menu_icons = ["🏠", "😃", "✈️", "🔢", "🖼️", "👶", "✍️", "🃏", "🚨", "✌️"]

    # Chọn chức năng từ thanh menu
    st.sidebar.markdown("<h2 style='font-size:24px; color: black;'>👉 Chọn chức năng 👈</h2>", unsafe_allow_html=True)
    choice = st.sidebar.selectbox("", menu, format_func=lambda x: menu_icons[menu.index(x)] + " " + x)

    if choice == "Trang chủ":
        display_home()
    elif choice == "Nhận dạng khuôn mặt":
        display_face_detection()
    elif choice == "Nhận dạng đối tượng":
        display_object_detection()
    elif choice == "Nhận dạng chữ số viết tay":
        display_handwritten_digit_recognition()
    elif choice == "Xử lý ảnh":
        display_image_processing()
    elif choice == "Nhận dạng tuổi - giới tính":
        display_face_Age_Gender()
    elif choice == "Nhận dạng chữ viết tay":
        hand_writting()
    elif choice == "Nhận dạng lá bài tây":
        cards()
    elif choice == "Phát hiện gian lận":
        exam_cheating()
    elif choice == "Nhận diện tay điều chỉnh âm lượng":
        hand_volumeAdjust()

if __name__ == "__main__":
    main()