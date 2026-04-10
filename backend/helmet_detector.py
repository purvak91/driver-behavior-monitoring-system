import cv2
import os

class HelmetDetector:
    def __init__(self):
        # We use a standard Face Cascade. 
        # Logic: If we clearly see a face on a rider, they are NOT wearing a full-face helmet.
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def is_wearing_helmet(self, person_crop) -> bool:
        """
        Takes an OpenCV BGR crop of the rider.
        Returns True if a helmet is assumed, False if definitely no helmet.
        """
        if person_crop is None or person_crop.size == 0:
            return True # Edge case: can't evaluate

        h, w = person_crop.shape[:2]
        
        # Focus heavily on the top 35% of the bounding box where the head should be
        head_roi_h = int(h * 0.35)
        if head_roi_h <= 10:
            return True
            
        head_crop = person_crop[0:head_roi_h, 0:w]
        
        # UPSCALING HACK: The video resolution is too low for the Haar Cascade to detect tiny faces.
        # We upscale the head region by 300% to simulate a high-res camera!
        upscaled = cv2.resize(head_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        # We also lower minNeighbors to 2 to be extremely aggressive at catching a face
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=2, 
            minSize=(20, 20)
        )
        
        # If any clear face is detected in the upper region, they aren't wearing a proper helmet!
        if len(faces) > 0:
            print(f"[HELMET] Found {len(faces)} face(s) -> NO HELMET!")
            return False
            
        return True
