from __future__ import annotations
import os
import time
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# MediaPipe Hands (0.10.x)
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# （将来の同期度計算で使う場合に備えて残置）

# -----------------------------
# Helpers
# -----------------------------
@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int

def get_video_meta(cap: cv2.VideoCapture) -> VideoMeta:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-6:
        fps = 30.0
    f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMeta(w, h, fps, f)

def aspect_resize(w: int, h: int, target_w: Optional[int]) -> Tuple[int, int]:
    if not target_w or target_w <= 0:
        return w, h
    s = target_w / float(w)
    return target_w, int(round(h * s))

def make_writer(path: str, size: Tuple[int, int], fps: float, use_mp4: bool) -> cv2.VideoWriter:
    """
    コーデック自動フォールバック:
      MP4: mp4v -> avc1 -> H264 -> (失敗時) AVI(XVID)
      AVI: XVID
    """
    candidates = [("XVID", ".avi")] if not use_mp4 else [("mp4v", ".mp4"), ("avc1", ".mp4"), ("H264", ".mp4")]
    last_err = None
    for fourcc_str, ext in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out_path = path if path.endswith(ext) else path + ext
        w = cv2.VideoWriter(out_path, fourcc, fps, size)
        if w.isOpened():
            return w
        last_err = f"FOURCC {fourcc_str} failed for {out_path}"
    # 最終フォールバック AVI
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path = path if path.endswith(".avi") else path + ".avi"
    w = cv2.VideoWriter(out_path, fourcc, fps, size)
    if w.isOpened():
        return w
    raise RuntimeError(f"Failed to open video writer. {last_err or ''} Try AVI or install codecs.")

# -----------------------------
# 書き出し（Handsのみ）
# -----------------------------
def process_video_hands(
    input_path: str,
    out_base_path: str,
    draw_landmarks: bool = True,
    max_num_hands: int = 2,
    det_conf: float = 0.5,
    track_conf: float = 0.5,
    output_width: Optional[int] = None,
    use_mp4: bool = True,
    update_progress=None,
) -> Tuple[str, VideoMeta]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")
    meta = get_video_meta(cap)
    out_w, out_h = aspect_resize(meta.width, meta.height, output_width)

    writer = make_writer(out_base_path, (out_w, out_h), meta.fps, use_mp4)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )

    idx = 0
    last = time.time()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if draw_landmarks and res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        bgr,
                        lm,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            if (bgr.shape[1], bgr.shape[0]) != (out_w, out_h):
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.write(bgr)
            idx += 1
            if update_progress and (time.time() - last) > 0.1 and meta.frame_count > 0:
                update_progress(min(idx / meta.frame_count, 1.0))
                last = time.time()
    finally:
        cap.release()
        writer.release()
        hands.close()

    # どちらが書かれたか判定
    mp4, avi = out_base_path + ".mp4", out_base_path + ".avi"
    final = mp4 if os.path.exists(mp4) else (avi if os.path.exists(avi) else out_base_path + (".mp4" if use_mp4 else ".avi"))
    return final, meta

# -----------------------------
# Streamlit UI（安定化版）
# -----------------------------
st.set_page_config(page_title="Hands限定：骨格検出＆同期度", layout="wide")
st.title("Hands限定 骨格ダウンロード")
st.caption("検出は手だけ（MediaPipe Hands）。CSVは出力しません。")

# プレビュー用のバイト列スロット（DOM安定のため固定）
if "vid_bytes" not in st.session_state:
    st.session_state["vid_bytes"] = None

with st.sidebar:
    st.header("書き出し設定（Hands）")
    draw_landmarks = st.checkbox("ランドマークを描画", value=True)
    output_width = st.number_input("出力幅(px)（空なら元サイズ）", min_value=0, max_value=4096, value=0, step=32)
    max_num_hands = st.slider("最大手数", 1, 2, 2)
    det_conf = st.slider("検出信頼度", 0.1, 0.9, 0.5, 0.05)
    track_conf = st.slider("追跡信頼度", 0.1, 0.9, 0.5, 0.05)
    container_fmt = st.selectbox("書き出しコンテナ", ["MP4 (mp4v)", "AVI (XVID)"])

# アップロード＆プレビュー（UploadedFileを直接st.videoに渡さない）
uploaded = st.file_uploader(
    "動画ファイルをアップロード (mp4/avi/mov/mkv)",
    type=["mp4", "mov", "avi", "mkv"],
    key="uploader_main",
)

col1, col2 = st.columns(2)
with col1:
    if uploaded is not None:
        st.session_state["vid_bytes"] = uploaded.getvalue()  # bytesに固定
    preview_slot = st.empty()  # 器を固定
    if st.session_state["vid_bytes"] is not None:
        preview_slot.video(st.session_state["vid_bytes"], autoplay=False)

# 実行ボタン（bytesの有無で制御）
run = st.button("この設定で処理する", type="primary", disabled=(st.session_state["vid_bytes"] is None))

if run and st.session_state["vid_bytes"] is not None:
    use_mp4 = container_fmt.startswith("MP4")
    try:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "input")
            # 保存はbytesから（seek不要で安全）
            with open(in_path, "wb") as f:
                f.write(st.session_state["vid_bytes"])

            base_out = os.path.join(td, "output")
            prog = st.progress(0.0, text="処理中…")
            def _update(p):
                prog.progress(p, text=f"処理中… {int(p * 100)}%")

            with st.spinner("MediaPipe Handsで推論中…"):
                out_path, meta = process_video_hands(
                    input_path=in_path,
                    out_base_path=base_out,
                    draw_landmarks=draw_landmarks,
                    max_num_hands=max_num_hands,
                    det_conf=det_conf,
                    track_conf=track_conf,
                    output_width=(output_width or None),
                    use_mp4=use_mp4,
                    update_progress=_update,
                )
            prog.empty()

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            st.success(
                f"完了：{os.path.basename(out_path)}  |  元: {meta.width}x{meta.height}@{meta.fps:.1f}fps  |  フレーム数: {meta.frame_count}"
            )
            col_v, col_d = st.columns([2, 1])
            with col_v:
                st.subheader("プレビュー")
                st.video(video_bytes)
            with col_d:
                st.subheader("ダウンロード")
                st.download_button(
                    label="動画をダウンロード",
                    data=video_bytes,
                    file_name=os.path.basename(out_path),
                    mime="video/mp4" if use_mp4 else "video/avi",
                )
    except Exception as e:
        st.error(
            f"エラー: {e}\n\n対処案:\n"
            "- '書き出しコンテナ' を 'AVI' に変更\n"
            "- 別の動画で試す / 出力幅を下げる\n"
            "- OpenCV/MediaPipe を更新: pip install -U opencv-python mediapipe\n"
            "- Windowsは“HEVC Video Extensions”導入も検討"
        )
else:
    st.info("動画をアップロードしてから『この設定で処理する』を押してください。")
