# Reward-Based Driver Behavior Monitoring System

An end-to-end AI-powered traffic surveillance pipeline using Computer Vision to detect red light violations, read license plates, and generate gamified driver safety scores.

## ⚠️ Development Rules (For Contributors)
During the creation of this project, we strictly adhered to the following principles:
- **Atomic Commits:** Features are encapsulated in fully-tested, functional groups before committing.
- **Time/FPS Constraints:** All deep learning evaluations (YOLO, OpenCV) are bounded to a maximum Frame-Per-Second (FPS) refresh rate (currently 5 FPS). This ensures the system acts exactly like a delayed CCTV camera without destroying CPU limits.

## System Architecture (All Phases Complete)

### Vision Pipeline (`backend/vision_pipeline.py`)
- **Real-Time Detection:** YOLOv8 Nano for detecting vehicles and pedestrians. Includes DeepSORT persistent object tracking (each vehicle gets a permanent ID).
- **ANPR Engine:** EasyOCR integration mapped to the bottom 50% region of vehicle bounding boxes for license plate recognition.
- **Image Upscaling + Enhancement:** 200% CUBIC interpolation upscale, CLAHE contrast enhancement, and sharpening kernel to extract plates from low-resolution CCTV/dashcam feeds.
- **Multi-line Plate Support:** EasyOCR `paragraph=True` mode stitches two-line plates into single readings.
- **Confidence Filtering:** Only plates with ≥0.75 OCR confidence and valid HSRP regex format are accepted.
- **State Caching:** Uses YOLO tracking IDs for one-time OCR per vehicle, conserving CPU.
- **API Telemetry Push:** Every detected plate is automatically POSTed to the FastAPI backend.

### Traffic Signal Detection (`backend/signal_detector.py`)
- **YOLO-Based Crop:** Detects traffic light bounding boxes using COCO class_id=9.
- **HSV Color Analysis:** Classifies signal state as RED, GREEN, YELLOW, or UNKNOWN.

### Manual Stop Line Calibration
- **Interactive Setup:** On startup, the first frame is displayed. User clicks 2 points to define the stop line at any angle.
- **Perspective-Correct:** Works for tilted/angled camera views — not restricted to horizontal lines.
- **Coordinate Scaling:** Supports 4K video with scaled display window.

### Vehicle Finite State Machine (`backend/vehicle_fsm.py`)
- **Per-Vehicle FSM:** States: `MOVING → STOPPED → VIOLATED → CLEARED`
- **Cross-Product Geometry:** Line crossing detected using `side(p) = sign((x2-x1)*(py-y1) - (y2-y1)*(px-x1))`. Violation fires when `side(prev) ≠ side(curr)` during RED signal.
- **ROI Filter:** Automatically ignores opposite-side traffic based on stop line position.
- **Violation Evidence:** Snapshot frames with stop line overlay saved to `backend/violations/`.

### Visual Output Color Guide
| Color | Thickness | Meaning |
|-------|-----------|---------|
| 🟢 Green | 2px | Vehicle moving normally |
| 🟡 Yellow | 2px | Vehicle stopped |
| 🔴 Red + Banner | 4px | RED LIGHT VIOLATION detected |
| 🟠 Orange | 3px | Previously violated, now cleared |

### Backend API (`backend/main.py`)
- **FastAPI** server with SQLite + SQLAlchemy database.
- `POST /api/telemetry` — Vision pipeline pushes plate detections.
- `POST /api/violation` — Report traffic violations with point deductions.
- `GET /api/leaderboard` — Ranked driver list by safety score.
- `GET /api/stats` — Aggregated dashboard statistics.
- `GET /api/recent-activity` — Latest tracking events.
- `WS /api/live-feed` — WebSocket for real-time dashboard streaming.

### Frontend Dashboard (`frontend/`)
- **React + Vite** with a dark neon glassmorphism theme.
- Real-time stat cards, gamified safety leaderboard, and live detection feed.
- WebSocket connection with automatic reconnection + polling fallback.

## How to Run

### 1. Start the Backend API
```bash
pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --reload --port 8000
```

### 2. Start the Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```

### 3. Run the Vision Pipeline
Place a traffic video inside `backend/sample_traffic_recorded/`, then:
```bash
cd backend
python vision_pipeline.py
```

**On startup:**
1. A window opens showing the first frame of the video.
2. **Click 2 points** on the road to define the stop line (e.g., along the zebra crossing).
3. The system begins processing. Vehicles crossing the stop line during a red signal are flagged with thick red bounding boxes and a `!! VIOLATION !!` banner.
4. Violation snapshots are saved to `backend/violations/`.
5. Press `q` to quit gracefully.

*Detected plates are automatically pushed to the API and appear on the dashboard in real-time.*