# Webcam Face Detection with OpenCV

## 1. Problem Statement

This project addresses a practical computer vision problem: detecting human faces from a live webcam feed in real time.

Face detection is often the first stage in larger applications such as attendance systems, smart camera tools, security monitoring, and human-computer interaction. The challenge is to build a solution that is fast, simple to run, and understandable for someone learning computer vision.

## 2. Why This Problem Matters

Real-time face detection matters for three main reasons:

1. It is a foundational skill in computer vision pipelines.
2. It demonstrates how image-processing concepts work under performance constraints.
3. It is directly useful in many beginner-to-intermediate projects.

This project is intentionally designed to be accessible: no GPU requirement, no large pretrained deep learning model download, and a straightforward setup with Python + OpenCV.

## 3. Project Goals

The goals were:

1. Build a working real-time face detector using a webcam.
2. Provide two ways to run it:
	- CLI application for direct OpenCV window use
	- Minimal browser frontend for easier interaction
3. Keep the code modular and easy to extend.
4. Expose key detection parameters so behavior can be tuned.
5. Document tradeoffs, challenges, and lessons learned.

## 4. Approach

### 4.1 Detection Method

The core detector uses OpenCV Haar Cascade face detection (`haarcascade_frontalface_default.xml`).

Pipeline per frame:

1. Read frame from webcam.
2. Convert frame to grayscale.
3. Run `detectMultiScale` with configurable parameters.
4. Draw bounding boxes around detected faces.
5. Overlay metadata such as face count and optional FPS.

### 4.2 Application Modes

This project supports two execution modes:

1. CLI Mode (`run.py`)
	- Direct OpenCV window
	- Keyboard controls (`q` to quit, `s` to save snapshot)
	- Useful for quick local testing

2. Web Mode (`web_app.py`)
	- Flask app serving a minimal frontend
	- Streams processed frames via MJPEG endpoint (`/video_feed`)
	- Useful for demonstration and easier UX

### 4.3 Parameterization

The detector behavior can be tuned through parameters such as:

- camera index
- scale factor
- min neighbors
- minimum face size
- FPS overlay toggle

These controls make the project adaptable to different lighting, camera quality, and scene complexity.

## 5. Key Design Decisions and Rationale

### Decision 1: Haar Cascades Instead of Deep Learning Models

Why:

1. Lower setup complexity.
2. Faster startup and lighter dependencies.
3. Good for educational demos and constrained environments.

Tradeoff:

- Less robust than modern CNN-based detectors in difficult angles/lighting.

### Decision 2: Keep CLI and Add a Minimal Web Frontend

Why:

1. CLI is best for direct debugging and frame-by-frame testing.
2. Browser UI improves accessibility for non-technical users.

Tradeoff:

- Slightly more code to maintain (streaming route + simple frontend assets).

### Decision 3: Expose Detection Hyperparameters to the User

Why:

1. Helps tune accuracy vs speed.
2. Makes the project more reusable across webcams and environments.

Tradeoff:

- More user-facing options means slightly more documentation burden.

## 6. Challenges Faced and How They Were Solved

### Challenge 1: Webcam Access Reliability

Issue:

- Webcam device indices can vary by machine.

Solution:

- Added support for configurable camera index and documented fallback behavior.

### Challenge 2: Balancing Detection Accuracy vs Performance

Issue:

- Aggressive detection settings can increase false positives or reduce frame rate.

Solution:

- Exposed `scale_factor`, `min_neighbors`, and `min_size` controls with practical defaults.

### Challenge 3: Browser Streaming with OpenCV Frames

Issue:

- Browsers do not natively render raw OpenCV frames.

Solution:

- Implemented MJPEG streaming endpoint in Flask that yields JPEG-encoded frames continuously.

## 7. What I Learned

Key takeaways from this implementation:

1. Real-time CV projects are often bottlenecked by I/O and preprocessing, not only detection logic.
2. Parameter defaults matter significantly for user experience.
3. A small frontend can dramatically improve usability without major architectural overhead.
4. Clear project documentation is as important as model logic for reproducibility.

## 8. Current Features

- Real-time webcam face detection
- Bounding boxes on faces
- Live face count overlay
- Optional FPS overlay
- Snapshot capture in CLI mode
- Minimal browser frontend
- Tunable detection parameters

## 9. Project Structure

```
.
├── .gitignore
├── CHANGELOG.md
├── README.md
├── requirements.txt
├── run.py
├── web_app.py
├── static/
│   └── style.css
├── templates/
│   └── index.html
└── src/
	 ├── __init__.py
	 └── face_detection.py
```

## 10. Setup and Run

### 10.1 Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 10.2 Run Browser Frontend (Recommended)

```bash
python web_app.py
```

Open: `http://127.0.0.1:5000`

### 10.3 Run CLI Mode

```bash
python run.py
```

Example with options:

```bash
python run.py --camera-index 0 --scale-factor 1.1 --min-neighbors 5 --min-size 40 --draw-fps
```

### 10.4 CLI Keyboard Controls

- `q`: quit
- `s`: save current frame snapshot

## 11. Limitations

1. Haar cascades can miss faces under strong pose changes or low light.
2. The project currently focuses on detection, not recognition/identity.
3. Single-camera stream only.

## 12. Future Improvements

1. Add deep learning detector option (e.g., SSD/ONNX) behind a flag.
2. Add recording mode for processed video output.
3. Add simple test scripts for frame-processing utilities.
4. Add Docker support for reproducible setup.

## 13. Commit History Plan (Minimum 5 Commits)

Suggested commit sequence:

1. `chore: initialize Python OpenCV project scaffold`
2. `feat: add real-time webcam face detection pipeline`
3. `feat: add minimal Flask frontend for browser streaming`
4. `docs: rewrite README as structured project report`
5. `chore: finalize project metadata and changelog`