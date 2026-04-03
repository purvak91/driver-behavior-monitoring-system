# Reward-Based Driver Behavior Monitoring System

An end-to-end traffic surveillance pipeline using Computer Vision to detect violations and generate dynamic scores for drivers to gamify road safety.

## Key Features
- **Real-Time Detection:** YOLOv8 Nano for detecting vehicles and pedestrians.
- **ANPR System:** EasyOCR integration with regional text filtering.
- **Rule Engine:** Detects violations like red-light jumping and zebra crossing infractions.
- **Gamified Scoring:** Penalties for violations and rewards for safe driving streaks.
- **Dashboard:** React-based web UI with visual proof of violations.

## Architecture
- **Inference/AI:** Python (OpenCV, Ultralytics, EasyOCR)
- **Backend:** FastAPI, SQLite (SQLAlchemy)
- **Frontend:** React, Vite, TailwindCSS
