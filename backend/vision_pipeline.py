import os
import cv2
import time
import re
import easyocr
from ultralytics import YOLO

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

    def extract_and_read_plate(self, vehicle_roi):
        """Finds plate in vehicle crop, runs OCR, and validates with Regex."""
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        
        # Deep Learning Trick: Upscale the tiny cropped region by 200% using CUBIC interpolation.
        # Because the video is low-res, EasyOCR doesn't possess enough raw physical pixels to guess letters.
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        results = self.reader.readtext(upscaled)
        
        for (bbox, text, prob) in results:
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # For low-res prototyping, we drop the regex length constraint down to 4 characters (partial plate reads).
            # We drop prob to 0.15 and simply mandate it MUST contain at least one digit!
            if len(clean_text) >= 4 and prob > 0.15:
                if any(char.isdigit() for char in clean_text):
                    return clean_text
        return None

    def run(self):
        print(f"Attempting to open video feed from: {self.video_source}")
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"> Error: Could not open video source '{self.video_source}'")
            return

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
                
                # YOLO draws its default boxes
                annotated_frame = results[0].plot()

                # Phase 3 Custom Logic: Intercept YOLO data for ANPR Extraction
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy()
                    
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        # Filter to vehicles mathematically (COCO dataset: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck)
                        if int(class_id) in [2, 3, 5, 7]:
                            x1, y1, x2, y2 = map(int, box)
                            
                            # If we haven't successfully OCR'd this vehicle ID yet...
                            if track_id not in self.plate_registry:
                                # Fix: Plates are almost always on the bottom half of the vehicle.
                                # Crop out the top 50% of the box to avoid reading side-ads on buses!
                                box_height = y2 - y1
                                vehicle_roi = frame[y1 + int(box_height * 0.5):y2, x1:x2]
                                if vehicle_roi.size > 0:
                                    plate_text = self.extract_and_read_plate(vehicle_roi)
                                    if plate_text:
                                        self.plate_registry[track_id] = plate_text
                                        print(f"New Plate Discovered! Vehicle ID #{track_id} -> {plate_text}")
                            
                            # If plate is in registry, permanently draw it above the vehicle
                            if track_id in self.plate_registry:
                                text_to_draw = f"PLATE: {self.plate_registry[track_id]}"
                                # Draw thick neon green text
                                cv2.putText(annotated_frame, text_to_draw, (x1, y1 - 40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

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

if __name__ == "__main__":
    pipeline = VisionPipeline(video_source="sample_traffic.mp4", target_fps=5)
    pipeline.run()
