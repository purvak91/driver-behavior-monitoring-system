# Phase 1: Environment Initialization & Tracker
- [x] Create `.gitignore` and `PROJECT_TRACKER.md`
- [x] Initialize Git repository
- [x] Setup Python backend structure & virtual environment
- [x] Setup React + Vite frontend template
- [x] Initial Git commit

# Phase 2: Video Ingestion & Basic AI
- [x] Implement video ingestion script (5 FPS limit)
- [x] Integrate YOLOv8 Nano inference
- [x] Integrate persistent Object ID tracking

# Phase 3: Advanced AI (ANPR & State Models)
- [x] EasyOCR integration for number plates
- [x] Regional Regex configuration & State caching
- [x] Confidence-based OCR filtering (≥0.75 threshold)
- [x] HSRP format validation with alphabet/digit correction
- [x] Image upscaling (200% CUBIC) + CLAHE contrast enhancement

# Phase 4: Core Logic & Scoring Engine
- [x] Traffic signal detection via YOLO (class_id=9) + HSV color analysis
- [x] Per-vehicle Finite State Machine (MOVING → STOPPED → VIOLATED → CLEARED)
- [x] Manual stop line calibration (2-point mouse click selection)
- [x] Cross-product based line crossing detection (orientation-agnostic)
- [x] ROI filtering to ignore opposite-side traffic
- [x] Violation frame snapshots saved to `backend/violations/`
- [x] SQLite Database models for Drivers and Violations
- [x] Gamified scoring logic implementation (safety scores, sightings)

# Phase 5: API & Frontend Dashboard
- [x] FastAPI Endpoints (telemetry, violations, leaderboard, stats, recent-activity)
- [x] WebSocket live-feed endpoint for real-time streaming
- [x] React UI (Dashboard with dark neon glassmorphism theme)
- [x] Real-time stat cards, gamified safety leaderboard
- [x] Telemetry auto-push from vision pipeline to API
