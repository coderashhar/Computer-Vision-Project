from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Webcam face detection using OpenCV")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Haar cascade scale factor (default: 1.1)",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Haar cascade min neighbors (default: 5)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=40,
        help="Minimum face size in pixels (default: 40)",
    )
    parser.add_argument(
        "--draw-fps",
        action="store_true",
        help="Draw FPS on the output frame",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="captures",
        help="Directory to save snapshots when pressing 's' (default: captures)",
    )
    return parser


def load_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
    return cascade


def run_face_detection(args: argparse.Namespace) -> None:
    face_cascade = load_face_cascade()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not access webcam at index {args.camera_index}. "
            "Try another index with --camera-index 1"
        )

    prev_time = time.time()
    print("Press 'q' to quit, 's' to save a snapshot.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(args.min_size, args.min_size),
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

        if args.draw_fps:
            current_time = time.time()
            delta = max(current_time - prev_time, 1e-9)
            fps = 1.0 / delta
            prev_time = current_time
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

        cv2.imshow("OpenCV Face Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            timestamp = int(time.time())
            snapshot_path = save_dir / f"snapshot_{timestamp}.jpg"
            cv2.imwrite(str(snapshot_path), frame)
            print(f"Saved snapshot: {snapshot_path}")

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_face_detection(args)


if __name__ == "__main__":
    main()
