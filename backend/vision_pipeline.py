import os
import cv2
import time
import re
import numpy as np
import easyocr
import requests
from ultralytics import YOLO

# Phase 4a: Import Signal Detector
from signal_detector import TrafficSignalDetector, SignalState
# Phase 4b: Import Vehicle FSM Manager
from vehicle_fsm import FSMManager, VehicleState

# Backend API URL — the FastAPI server must be running
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

class VisionPipeline:
    def __init__(self, video_source="sample_traffic.mp4", target_fps=5):
        # Resolve path relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_source = os.path.join(script_dir, video_source)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        print("Loading YOLOv8 Nano model...")
        self.model = YOLO('yolov8n.pt') 

        print("Initializing EasyOCR (this may take a few moments)...")
        # Initialize EasyOCR (gpu=False so it runs on your Ryzen CPU)
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # We use a standard OpenCV Haar cascade to find the physical license plate bounding box
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_russian_plate_number.xml')
        self.plate_cascade = cv2.CascadeClassifier(cascade_path)
        
        # State Management: We do not process OCR on the same car twice to save CPU!
        # Dictionary format: { track_id : "DL8CNA1234" }
        self.plate_registry = {}

        # Phase 4a: Initialize Traffic Signal Detector
        # Using pure visual detection from dynamic YOLO traffic light cropped bounding box
        self.signal_detector = TrafficSignalDetector()

        # Phase 4b: Initialize FSM Manager
        # We start with a placeholder, the user will configure it
        self.stop_line = ((0, 1500), (3840, 1500))
        self.fsm_manager = FSMManager(stop_line=self.stop_line)
        
        # ROI filter: only process vehicles whose centroid Y > this threshold.
        # Vehicles above this line are on the OPPOSITE side of the intersection.
        # This is set dynamically as a fraction of frame height on the first frame.
        self.roi_y_min = None

    def correct_plate_format(self, plate_text: str) -> str:
        """Attempts to fix common OCR confusions based on the LLDDLLDDDD format."""
        # We only apply strict position correction if it's exactly 10 characters
        if len(plate_text) != 10:
            return plate_text
            
        char_to_digit = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'A': '4'}
        digit_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '4': 'A', '7': 'T'}
        
        # Expected format: L L D D L L D D D D
        expected_types = ['L', 'L', 'D', 'D', 'L', 'L', 'D', 'D', 'D', 'D']
        
        corrected = ""
        for i, char in enumerate(plate_text):
            expected = expected_types[i]
            if expected == 'L' and char.isdigit():
                corrected += digit_to_char.get(char, char)
            elif expected == 'D' and char.isalpha():
                corrected += char_to_digit.get(char, char)
            else:
                corrected += char
                
        return corrected

    def extract_and_read_plate(self, vehicle_roi):
        """Finds plate in vehicle crop, runs OCR, and validates with Regex."""
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        
        # Deep Learning Trick: Upscale the tiny cropped region by 200% using CUBIC interpolation.
        # Because the video is low-res, EasyOCR doesn't possess enough raw physical pixels to guess letters.
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        # Blur Fix: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(upscaled)
        
        # Blur Fix: Sharpening kernel
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        
        # Multiline Fix: manual combination to retain confidence
        # Alphabet Fix: allowlist
        results = self.reader.readtext(
            sharpened, 
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        
        if not results:
            return None, 0.0, False
            
        combined_text = ""
        total_prob = 0.0
        
        for result in results:
            bbox, text, prob = result
            combined_text += text
            total_prob += prob

        avg_prob = total_prob / len(results)
        clean_text = re.sub(r'[^A-Z0-9]', '', combined_text.upper())
        
        # Heuristic correction based on standard LLDDLLDDDD format
        clean_text = self.correct_plate_format(clean_text)
        
        # HSRP Regex: Indian Plate Format LLDDLLDDDD (e.g., MH12AB1234)
        hsrp_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$')
        is_hsrp = bool(hsrp_pattern.match(clean_text))
        
        # For low-res prototyping, we drop the regex strictness down to 6 characters (partial plate reads).
        if len(clean_text) >= 6:
            if any(char.isdigit() for char in clean_text):
                return clean_text, avg_prob, is_hsrp
                
        return None, 0.0, False

    def get_manual_stop_line(self, cap):
        """Pauses on the first frame to allow user to draw the stop line with 2 clicks."""
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame for calibration.")
            return ((0, 1500), (3840, 1500))
            
        height, width = frame.shape[:2]
        
        display_frame = frame.copy()
        scale = 1.0
        if width > 1280:
            scale = 1280 / width
            display_frame = cv2.resize(display_frame, (1280, int(height * scale)))
            
        window_name = "Select 2 Points for Stop Line (Press ESC to keep default)"
        cv2.namedWindow(window_name)
        
        pts = []
        
        def select_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(pts) < 2:
                    pts.append((x, y))
                    cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                    if len(pts) >= 2:
                        cv2.line(display_frame, pts[0], pts[1], (0, 0, 255), 2)
                    cv2.imshow(window_name, display_frame)

        cv2.setMouseCallback(window_name, select_point)
        cv2.imshow(window_name, display_frame)
        
        print("************************************************************")
        print(">> PLEASE CLICK EXACTLY 2 POINTS TO DEFINE THE STOP LINE <<")
        print("   (Click on the window that opened. Press ESC to cancel.)")
        print("************************************************************")
        
        while len(pts) < 2:
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC
                break
                
        cv2.destroyWindow(window_name)
        
        if len(pts) == 2:
            # Scale back to original resolution
            orig_pts = []
            for p in pts:
                orig_pts.append((int(p[0] / scale), int(p[1] / scale)))
                
            # Assure line goes somewhat left-to-right to make side math consistent for vehicles
            if orig_pts[0][0] > orig_pts[1][0]:
                orig_pts = [orig_pts[1], orig_pts[0]]
                
            print(f"User selected stop line: {orig_pts}")
            return tuple(orig_pts)
            
        print("Selection cancelled, falling back to default line.")
        return ((0, int(height*0.6)), (width, int(height*0.6)))

    def run(self):
        print(f"Attempting to open video feed from: {self.video_source}")
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"> Error: Could not open video source '{self.video_source}'")
            return

        # Prompt user to select line before looping
        self.stop_line = self.get_manual_stop_line(cap)
        self.fsm_manager.set_stop_line(self.stop_line)

        print(f"Successfully started video processing... Limited to {self.target_fps} FPS.")
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream reached.")
                break

            current_time = time.time()
            if (current_time - prev_time) >= self.frame_interval:
                prev_time = current_time
                
                # Run YOLO inference WITH TRACKING
                results = self.model.track(frame, persist=True, conf=0.3, verbose=False)
                
                # Use a clean copy of the frame so we don't draw YOLO's default class/prob labels
                annotated_frame = frame.copy()
                
                frame_height, frame_width = frame.shape[:2]
                
                # --- Set ROI boundary on first frame based on Stop Line ---
                # This prevents opposite-side traffic from getting tracked
                if self.roi_y_min is None:
                    # Vehicles above the line (lower Y coordinate) by some margin are ignored
                    min_y_line = min(self.stop_line[0][1], self.stop_line[1][1])
                    self.roi_y_min = int(min_y_line * 0.7) # Give 30% margin above the stop line
                    print(f"[ROI] Set opposite-side filter: ignoring vehicles with centroid Y < {self.roi_y_min}")

                # Phase 4a: Detect traffic signal dynamically using YOLO (class_id == 9)
                signal_crop = None
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy()
                    for box, cls in zip(boxes, class_ids):
                        if int(cls) == 9: # Traffic Light class in COCO
                            tx1, ty1, tx2, ty2 = map(int, box)
                            signal_crop = frame[max(0, ty1):ty2, max(0, tx1):tx2]
                            
                            # Give a visual indication that YOLO found the traffic light
                            cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (255, 0, 255), 2)
                            cv2.putText(annotated_frame, "YOLO SIGNAL", (tx1, max(ty1 - 5, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                            break  # Use the first detected traffic light
                            
                signal_state = self.signal_detector.detect_from_crop(signal_crop)
                
                # Draw Phase 4b manual stop-line vividly
                (sx1, sy1), (sx2, sy2) = self.stop_line
                cv2.line(annotated_frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
                # Label it in the middle of the line
                mid_x, mid_y = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                cv2.putText(annotated_frame, "USER STOP LINE", (mid_x - 50, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw ROI boundary line (faint cyan) to show the opposite-side filter
                cv2.line(annotated_frame, (0, self.roi_y_min), (frame_width, self.roi_y_min), (255, 200, 0), 1)
                cv2.putText(annotated_frame, "-- OUR LANE BELOW --", (20, self.roi_y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                # Draw Phase 4a signal
                signal_color = (0, 255, 0) if signal_state == SignalState.GREEN else (0, 0, 255) if signal_state == SignalState.RED else (0, 255, 255)
                cv2.putText(annotated_frame, f"SIGNAL: {signal_state.value}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, signal_color, 2)

                # Collect tracking data for FSM (Phase 4b)
                current_frame_vehicles = {}

                # Phase 3 Custom Logic: Intercept YOLO data for ANPR Extraction
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy()
                    
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        # Filter to vehicles mathematically
                        if int(class_id) in [1, 2, 3, 5, 7]:
                            x1, y1, x2, y2 = map(int, box)
                            
                            # --- ROI FILTER: Skip vehicles from the opposite side ---
                            centroid_y = (y1 + y2) // 2
                            if centroid_y < self.roi_y_min:
                                # This vehicle is in the upper portion (opposite side), skip it entirely
                                continue
                            
                            # Cache info for FSM later
                            best_plate = self.plate_registry.get(track_id, f"UNK-{track_id}")
                            current_frame_vehicles[track_id] = {'plate': best_plate, 'bbox': (x1, y1, x2, y2)}
                            
                            # If we haven't successfully OCR'd this vehicle ID yet...
                            if track_id not in self.plate_registry:
                                box_width = x2 - x1
                                box_height = y2 - y1
                                
                                # Optimization & Fix: Only attempt OCR if the vehicle is close enough to the camera
                                # (e.g., bounding box is sufficiently large)
                                if box_width > 120 and box_height > 120:
                                    # Crop out the top 50% of the box to avoid reading side-ads on buses!
                                    vehicle_roi = frame[y1 + int(box_height * 0.5):y2, x1:x2]
                                    if vehicle_roi.size > 0:
                                        plate_text, conf, is_hsrp = self.extract_and_read_plate(vehicle_roi)
                                        if plate_text:
                                            if conf >= 0.75 and is_hsrp:
                                                self.plate_registry[track_id] = plate_text
                                                print(f"New Plate Discovered! Vehicle ID #{track_id} -> {plate_text} (Conf: {conf:.2f})")
                                                # Phase 4: Push telemetry to the backend API
                                                self._push_telemetry(plate_text, int(track_id), conf)
                                            else:
                                                reason = "Low Confidence" if conf < 0.75 else "Regex Mismatch"
                                                print(f"Rejected Plate: {plate_text} | Conf: {conf:.2f} | Reason: {reason}")
                            
                            # If plate is in registry, permanently draw it above the vehicle
                            if track_id in self.plate_registry:
                                text_to_draw = f"PLATE: {self.plate_registry[track_id]}"
                                # Draw thick neon green text
                                cv2.putText(annotated_frame, text_to_draw, (x1, y1 - 40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

                # --- PHASE 4b Execution ---
                violations = self.fsm_manager.update(current_frame_vehicles, frame, signal_state)
                
                # Process new violations
                for viol in violations:
                    print(f"!!! RED LIGHT VIOLATION DETECTED !!! Vehicle {viol['plate']} | Timestamp: {viol['timestamp']}")
                    print(f"Saved violation frame: {viol['frame_path']}")
                    # Telemetry push for violation
                    # self._push_telemetry(viol['plate'], viol['track_id'], 1.0, violation="RED_LIGHT")

                # Enhance visual output with custom bounding boxes, IDs, and FSM active states
                violation_count = 0
                for track_id, info in current_frame_vehicles.items():
                    x1, y1, x2, y2 = info['bbox']
                    plate_label = info.get('plate', f"UNK-{track_id}")
                    
                    box_color = (0, 255, 0) # Green for regular/moving
                    box_thickness = 2
                    state_label = "MOVING"
                    
                    if track_id in self.fsm_manager.vehicles:
                        fsm_state = self.fsm_manager.vehicles[track_id].state
                        state_label = fsm_state.value
                        
                        if fsm_state == VehicleState.STOPPED:
                            box_color = (0, 255, 255) # Yellow
                        elif fsm_state == VehicleState.VIOLATED:
                            box_color = (0, 0, 255) # Bright Red
                            box_thickness = 4
                            violation_count += 1
                            
                            # Draw a filled red banner at the top of the box
                            banner_h = 30
                            cv2.rectangle(annotated_frame, (x1, y1 - banner_h), (x2, y1), (0, 0, 255), -1)
                            cv2.putText(annotated_frame, "!! VIOLATION !!", (x1 + 5, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        elif fsm_state == VehicleState.CLEARED:
                            box_color = (0, 165, 255) # Orange for cleared
                            box_thickness = 3
                            violation_count += 1
                            
                    # Draw custom Vehicle Box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, box_thickness)
                    
                    # Display vehicle ID + state + plate
                    label = f"ID:{track_id} [{state_label}]"
                    cv2.putText(annotated_frame, label, (x1, max(y1 - 35, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    cv2.putText(annotated_frame, plate_label, (x1, max(y1 - 15, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show total violation counter on screen
                total_violations = sum(1 for v in self.fsm_manager.vehicles.values() 
                                       if v.state in [VehicleState.VIOLATED, VehicleState.CLEARED])
                cv2.putText(annotated_frame, f"VIOLATIONS: {total_violations}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # --------------------------

                height, width, _ = annotated_frame.shape
                if width > 1280:
                    scale = 1280 / width
                    annotated_frame = cv2.resize(annotated_frame, (1280, int(height * scale)))

                cv2.imshow("Reward-Based Driver Behavior Monitoring System - Core Vision API", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Manual quit triggered.")
                break

        cap.release()
        cv2.destroyAllWindows()

    def _push_telemetry(self, plate_number: str, track_id: int, confidence: float):
        """Non-blocking POST to the backend API. Fails silently if server is offline."""
        try:
            resp = requests.post(
                f"{API_URL}/api/telemetry",
                json={"plate_number": plate_number, "track_id": track_id, "confidence": confidence},
                timeout=2,
            )
            if resp.status_code == 200:
                data = resp.json()
                print(f"  -> API Synced: {data.get('plate_number')} | Sightings: {data.get('total_sightings')} | Score: {data.get('safety_score')}")
            else:
                print(f"  -> API Warning: status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print("  -> API Offline: Telemetry cached locally only.")
        except Exception as e:
            print(f"  -> API Error: {e}")


if __name__ == "__main__":
    pipeline = VisionPipeline(
        video_source=os.path.join("sample_traffic_recorded", "2026.mp4"),
        target_fps=5
    )
    pipeline.run()
