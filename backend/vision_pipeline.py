import os
import cv2
import time
from ultralytics import YOLO

class VisionPipeline:
    def __init__(self, video_source="sample_traffic.mp4", target_fps=5):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_source = os.path.join(script_dir, video_source)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Load the YOLOv8 Nano model.
        # Note: Upon first run, it will automatically download the "yolov8n.pt" file (approx 6 MB).
        print("Loading YOLOv8 Nano model...")
        self.model = YOLO('yolov8n.pt') 

    def run(self):
        print(f"Attempting to open video feed from: {self.video_source}")
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"> Error: Could not open video source '{self.video_source}'")
            print("> Action required: Please place an MP4 file named 'sample_traffic.mp4' in this folder and rerun.")
            return

        print(f"Successfully started video processing... Limited to {self.target_fps} FPS.")
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream reached.")
                break

            current_time = time.time()
            # Downsample: Only process frame if enough time has passed (to hit target FPS)
            if (current_time - prev_time) >= self.frame_interval:
                prev_time = current_time
                
                # 1. Run YOLO inference on the frame WITH TRACKING!
                # persist=True gives the same car the same ID across multiple frames.
                results = self.model.track(frame, persist=True, conf=0.3, verbose=False)
                
                # 2. Draw the bounding boxes and labels onto the image 
                annotated_frame = results[0].plot()

                # 3. Scale down the display window (otherwise 4K dashcam video will blow up screen)
                height, width, _ = annotated_frame.shape
                # If width is greater than 1280, resize display
                if width > 1280:
                    scale = 1280 / width
                    annotated_frame = cv2.resize(annotated_frame, (1280, int(height * scale)))

                # 4. Render to OpenCV window
                cv2.imshow("Reward-Based Driver Behavior Monitoring System - Core Vision API", annotated_frame)

            # Hit 'q' on keyboard to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Manual quit triggered.")
                break

        # Cleanup Memory
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test script: points defaults to "sample_traffic.mp4"
    pipeline = VisionPipeline(video_source="sample_traffic.mp4", target_fps=5)
    pipeline.run()
