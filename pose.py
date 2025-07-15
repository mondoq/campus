import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

st.set_page_config(page_title="てのほねけんしゅつアプリ", layout="centered")

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# タイトルと説明
st.markdown("<h1 style='color: #4D90FE;'>🖐️ てのほねを みてみよう！</h1>", unsafe_allow_html=True)
st.write("えいぞうや しゃしんを アップしてね。ての ほねを かくしてくれるよ！")

# アップロード
file = st.file_uploader("ファイルを えらんでね（しゃしん or どうが）", type=["jpg", "jpeg", "png", "mp4"])

# 検出関数
def detect_hand(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
    return image

# 処理
if file is not None:
    file_type = file.type
    if "image" in file_type:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        output_image = detect_hand(image)
        st.image(output_image, channels="BGR", caption="✨ これが きみの てのほね だよ！")
        st.success("すごいね！ての ほねが みえたよ 👏")

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        # 動画情報
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 出力動画の準備
        output_path = os.path.join(tempfile.gettempdir(), "output_hand_pose.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # ウォームアップ検出（最初のフレーム）
        ret, frame = cap.read()
        if ret:
            with mp_hands.Hands(static_image_mode=True) as warmup:
                _ = warmup.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                out.write(frame)

        cap.release()
        out.release()

        st.success("✨ ての ほねを けんしゅつしたよ！すごいね！")

        # ダウンロードボタン
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 けんしゅつしたどうがを ダウンロード",
                data=f,
                file_name="hand_pose_output.mp4",
                mime="video/mp4"
            )


