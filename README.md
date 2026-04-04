# Reward-Based Driver Behavior Monitoring System

An end-to-end traffic surveillance pipeline using Computer Vision to detect violations and generate dynamic scores for drivers to gamify road safety.

## Development Rules
During the creation of this project, we adhered to the following principles:
- **Time/FPS Constraints:** All deep learning evaluations (YOLO, OpenCV) are bounded to a maximum Frame-Per-Second (FPS) refresh rate. This ensures the system acts exactly like a delayed CCTV camera without destroying CPU limits.

## Key Features
- **Real-Time Detection:** YOLOv8 Nano for detecting vehicles and pedestrians.
- **ANPR System:** EasyOCR integration with regional text filtering.
- **Rule Engine:** Detects violations like red-light jumping and zebra crossing infractions.
- **Gamified Scoring:** Penalties for violations and rewards for safe driving streaks.
- **Dashboard:** React-based web UI with visual proof of violations.

## Technologies Used
- **Inference/AI:** Python (OpenCV, Ultralytics, YOLOv8n)
- **Backend API:** Python FastAPI, SQLAlchemy (SQLite Database)
- **Frontend UI:** React, Vite, TailwindCSS

---

## How to Run the Vision Pipeline (Phase 2 Test)

This component simulates analyzing a live surveillance camera feed. 
1. Place a testing video clip as `sample_traffic.mp4` directly inside the `backend/` directory.
2. Activate your virtual environment: `.\.venv\Scripts\activate`
3. Install dependencies: `pip install -r backend\requirements.txt`
4. Run the deep learning node:
```bash
python backend\vision_pipeline.py
```
Press `q` within the live OpenCV popup window to terminate the AI gracefully.