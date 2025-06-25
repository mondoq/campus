import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np

# MediaPipeのセットアップ
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image

def main():
    st.title("骨格検出アプリ")
    st.write("画像または動画をアップロードして、骨格を検出します。")

    file = st.file_uploader("画像または動画を選択", type=["jpg", "png", "jpeg", "mp4"])

    if file is not None:
        file_type = file.type
        if "image" in file_type:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            output_image = detect_pose(image)
            st.image(output_image, channels="BGR", caption="骨格検出結果")

        elif "video" in file_type:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            with mp_pose.Pose() as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    stframe.image(frame, channels="BGR")
            cap.release()

if __name__ == "__main__":
    main()
