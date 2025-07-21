import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# ページ設定
st.set_page_config(page_title="てのほねけんしゅつアプリ", layout="centered")

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# タイトル
st.markdown("<h1 style='color: #4D90FE;'>🖐️ てのほねを みてみよう！</h1>", unsafe_allow_html=True)
st.write("えいぞうや しゃしんを アップしてね。")

# 手の検出（画像用）
def detect_hand_image(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return image

# 手の検出（動画用・軽量化）
def detect_hand_video(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = 640, 360  # ✅ 解像度を固定して軽量化

    output_path = os.path.join(tempfile.gettempdir(), "output_hand_pose.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    stframe = st.empty()

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (w, h))  # ✅ 解像度を縮小

            if frame_idx % 2 == 0:  # ✅ 1フレームおきに推論
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()
    return output_path

# ファイルアップロード
file = st.file_uploader("ファイルを えらんでね（しゃしん or どうが）", type=["jpg", "jpeg", "png", "mp4"])

if file is not None:
    file_type = file.type

    if "image" in file_type:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            output_image = detect_hand_image(image)
            st.image(output_image, channels="BGR", caption="✨ これが きみの てのほね だよ！")
            st.success("すごいね！ての ほねが みえたよ 👏")
        else:
            st.error("画像の読み込みに失敗しました。")

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(file.read())
        tfile.flush()

        st.info("🎞️ 動画を処理中です。しばらくお待ちください...")
        output_video_path = detect_hand_video(tfile.name)

        st.success("✨ ての ほねを けんしゅつしたよ！すごいね！")

        with open(output_video_path, "rb") as f:
            st.download_button(
                label="📥 けんしゅつしたどうがを ダウンロード",
                data=f,
                file_name="hand_pose_output.mp4",
                mime="video/mp4"
            )
    else:
        st.error("対応していないファイル形式です。")
        
