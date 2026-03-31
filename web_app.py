from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from src.face_detection import load_face_cascade

app = Flask(__name__)

_latest_jpeg: bytes | None = None
_frame_lock = threading.Lock()


def _set_latest_jpeg(frame_jpeg: bytes) -> None:
    global _latest_jpeg
    with _frame_lock:
        _latest_jpeg = frame_jpeg


def _get_latest_jpeg() -> bytes | None:
    with _frame_lock:
        return _latest_jpeg


def _get_int_param(name: str, default: int, min_value: int = 0) -> int:
    raw = request.args.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, min_value)


def _get_float_param(name: str, default: float, min_value: float = 0.1) -> float:
    raw = request.args.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, min_value)


def generate_frames(
    camera_index: int,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    draw_fps: bool,
) -> Iterator[bytes]:
    face_cascade = load_face_cascade()
    cap = cv2.VideoCapture(camera_index)
    previous_time = time.time()

    if not cap.isOpened():
        frame = 255 * np.ones((360, 640, 3), dtype="uint8")
        cv2.putText(
            frame,
            "Could not open webcam.",
            (50, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Try camera index {camera_index + 1}.",
            (50, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 220),
            2,
            cv2.LINE_AA,
        )
        success, buffer = cv2.imencode(".jpg", frame)
        if success:
            _set_latest_jpeg(buffer.tobytes())
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_size, min_size),
            )

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 70), 2)

            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (20, 220, 255),
                2,
                cv2.LINE_AA,
            )

            if draw_fps:
                current_time = time.time()
                delta = max(current_time - previous_time, 1e-9)
                fps = 1.0 / delta
                previous_time = current_time
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            encoded, buffer = cv2.imencode(".jpg", frame)
            if not encoded:
                continue

            frame_jpeg = buffer.tobytes()
            _set_latest_jpeg(frame_jpeg)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_jpeg + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/video_feed")
def video_feed() -> Response:
    camera_index = _get_int_param("camera_index", 0, min_value=0)
    scale_factor = _get_float_param("scale_factor", 1.1, min_value=1.01)
    min_neighbors = _get_int_param("min_neighbors", 5, min_value=1)
    min_size = _get_int_param("min_size", 40, min_value=10)
    draw_fps = request.args.get("draw_fps", "0") == "1"

    return Response(
        generate_frames(
            camera_index=camera_index,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=min_size,
            draw_fps=draw_fps,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/health")
def health() -> Response:
    return jsonify({"status": "ok", "service": "face-detection-web"})


@app.post("/snapshot")
def snapshot() -> Response:
    frame_jpeg = _get_latest_jpeg()
    if frame_jpeg is None:
        return jsonify({"ok": False, "error": "No frame available yet. Start stream first."}), 400

    save_dir = Path("captures")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = save_dir / f"web_snapshot_{timestamp}.jpg"
    snapshot_path.write_bytes(frame_jpeg)

    return jsonify({"ok": True, "path": str(snapshot_path)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
