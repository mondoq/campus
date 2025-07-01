import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_hand(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2,min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
    return image

file = st.file_uploader("画像または動画をアップロード", type=["jpg", "jpeg", "png", "mp4"])

if file is not None:
    file_type = file.type
    if "image" in file_type:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        output_image = detect_hand(image)
        st.image(output_image, channels="BGR", caption="手の骨格検出結果")

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                else:
                    print("手を検出できませんでした")

                # フレームを表示
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

