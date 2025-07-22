import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# ページ設定
st.set_page_config(page_title="おりがみうごきよそくアプリ", layout="centered")

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ヘッダー
st.markdown("<h1 style='color: #FF69B4;'>🧠 おりがみうごきよそくアプリ</h1>", unsafe_allow_html=True)
st.write("📸 おりがみをしている どうがをアップしてね。つぎの てのうごきが いつかをよそくするよ！")

# 動き検出
def detect_movement_timing(landmarks_list, fps, threshold=0.03):
    movement_times = []
    prev_y = None

    for i, lm in enumerate(landmarks_list):
        if "RIGHT_WRIST" in lm:
            curr_y = lm["RIGHT_WRIST"].y
            if prev_y is not None:
                dy = abs(curr_y - prev_y)
                if dy > threshold:
                    movement_times.append(i / fps)
            prev_y = curr_y

    return movement_times

# 予測タイミング
def predict_next_time(movement_times):
    if len(movement_times) < 2:
        return None
    intervals = [t2 - t1 for t1, t2 in zip(movement_times, movement_times[1:])]
    avg_interval = sum(intervals) / len(intervals)
    return movement_times[-1] + avg_interval

# 骨格検出と動画出力
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = 640, 360

    landmarks_list = []
    output_path = os.path.join(tempfile.gettempdir(), "origami_movement_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    stframe = st.empty()

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (w, h))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            landmarks = {}
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for idx, lm in enumerate(result.pose_landmarks.landmark):
                    landmarks[mp_pose.PoseLandmark(idx).name] = lm

            landmarks_list.append(landmarks)
            out.write(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    out.release()
    return output_path, landmarks_list, fps

# ファイルアップロード
file = st.file_uploader("📥 どうがファイルを アップロードしてね（mp4形式）", type=["mp4"])

if file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(file.read())
    tfile.flush()

    st.info("🔍 どうがをけんしゅつちゅう... しばらくまってね！")
    output_path, landmarks_list, fps = process_video(tfile.name)

    st.success("✅ けんしゅつできたよ！")

     #動き・予測
    movement_times = detect_movement_timing(landmarks_list, fps)
    predicted_time = predict_next_time(movement_times)

    if movement_times:
        st.markdown(f"<h3>📍 {len(movement_times)} かい うごきがあったよ！</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>うごいたじかん：{', '.join(f'{t:.1f}s' for t in movement_times)}</h4>", unsafe_allow_html=True)

    if predicted_time:
       st.markdown(f"<h2 style='color: orange;'>🔮 つぎの てのうごきは {predicted_time:.1f} びょうごろかも！</h2>", unsafe_allow_html=True)
    else:
        st.warning("うごきがすくなくて、つぎのよそくがむずかしかったよ。")

    st.video(output_path)
    with open(output_path, "rb") as f:
        st.download_button("📥 ダウンロードする", f, "origami_movement_output.mp4", "video/mp4")

