# Reward-Based Driver Behavior Monitoring System

An end-to-end traffic surveillance pipeline using Computer Vision to detect violations and generate dynamic scores for drivers to gamify road safety.

## ⚠️ Development Rules (For Contributors)
During the creation of this project, we strictly adhered to the following principles:
- **Atomic Commits:** Features are encapsulated in fully-tested, functional groups before committing.
- **Time/FPS Constraints:** All deep learning evaluations (YOLO, OpenCV) are bounded to a maximum Frame-Per-Second (FPS) refresh rate (currently 5 FPS). This ensures the system acts exactly like a delayed CCTV camera without destroying CPU limits.

## Current System Architecture (Phase 3 Complete)
The project currently has a fully functional Python Vision Pipeline `vision_pipeline.py`.
- **Real-Time Detection:** YOLOv8 Nano for detecting vehicles and pedestrians. Includes `DeepSORT` persistent object tracking (each vehicle gets a permanent ID).
- **ANPR Engine:** EasyOCR integration mapped specifically to the bottom 50% region of cars to read license plates.
- **Image Upscaling:** The OCR pipeline automatically applies a 200% CUBIC interpolation upscale to realistically extract plates from low-resolution CCTV/dashcam feeds.
- **State Caching:** Uses YOLO's tracking IDs to log a plate reading. Once a vehicle drops a valid plate reading, OCR is mathematically shut off for that specific car to conserve CPU.
- **Regional text filtering:** Uses Regular Expressions to look for alphanumeric patterns and filters out advertisements. Currently tuned for Indian HSRP (High Security Registration Plate) style formats.

## How to Run the Vision Pipeline 

This component simulates analyzing a live surveillance camera feed. 
1. Place a testing video clip as `sample_traffic.mp4` directly inside the `backend/` directory.
2. Initialize your Python virtual environment and activate it.
3. Install dependencies: 
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Run the deep learning node:
   ```bash
   python backend/vision_pipeline.py
   ```
*Press `q` within the live OpenCV popup window to terminate the AI gracefully. When a plate is successfully detected, OCR shuts off and the plate string maps over the vehicle bounding box permanently.*