import cv2
import numpy as np
import time
from enum import Enum
from typing import List, Tuple

class SignalState(Enum):
    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    UNKNOWN = "UNKNOWN"

class TrafficSignalDetector:
    def __init__(self, roi_polygon: List[Tuple[int, int]] = None):
        """
        roi_polygon: Optional. No longer required if using YOLO dynamic cropping!
        """
        if roi_polygon is not None:
            self.roi_polygon = np.array(roi_polygon, dtype=np.int32)
        else:
            self.roi_polygon = np.array([])
            
        self.state_buffer = []  # Stores the last 5 frames' states
        self.buffer_size = 5
        self.current_state = SignalState.UNKNOWN

    def update_roi(self, new_roi_polygon: List[Tuple[int, int]]):
        self.roi_polygon = np.array(new_roi_polygon, dtype=np.int32)

    def _get_dominant_state(self, new_state: SignalState) -> SignalState:
        self.state_buffer.append(new_state)
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)

        counts = {state: self.state_buffer.count(state) for state in SignalState}
        
        # Majority rule: need 4/5 votes to change state
        for state, count in counts.items():
            if count >= 4:
                self.current_state = state   # ← WRITE it here
                return self.current_state
        
        # Fallback: return most frequent in buffer (not frozen old state)
        best = max(counts, key=lambda s: counts[s])
        return best

    def detect(self, frame: np.ndarray) -> SignalState:
        """
        Crops the ROI from the frame and uses HSV masking to find the active signal.
        """
        if self.roi_polygon.size == 0 or frame is None:
            return self._get_dominant_state(SignalState.UNKNOWN)

        # Create a mask for the polygon ROI
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_polygon], 255)
        
        # Extract the region
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to HSV
        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        
        # Defined HSV ranges
        # Red: H in [0-10] or [160-179], S > 100, V > 100
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        
        # Green: H in [40-90], S > 80, V > 80
        lower_green = np.array([40, 80, 80])
        upper_green = np.array([90, 255, 255])
        
        # Yellow: H in [15-35], S > 100, V > 100
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Create masks for each color
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Count non-zero pixels for each
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        
        # Determine raw state based on highest pixel count
        max_pixels = max(red_pixels, green_pixels, yellow_pixels)
        
        raw_state = SignalState.UNKNOWN
        if max_pixels > 10:  # Threshold to ignore noise
            if max_pixels == red_pixels:
                raw_state = SignalState.RED
            elif max_pixels == green_pixels:
                raw_state = SignalState.GREEN
            else:
                raw_state = SignalState.YELLOW
                
        self.current_state = self._get_dominant_state(raw_state)
        return self.current_state

    def detect_from_crop(self, crop: np.ndarray) -> SignalState:
        """
        Uses HSV masking on a direct YOLO bounding box crop to find the active signal.
        """
        # If no crop or very small, default to RED as requested
        if crop is None or crop.size == 0:
            return self._get_dominant_state(SignalState.RED)
            
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Red: Broadened to H= [0-30] (which includes orange/yellow typical of washed-out red lights) 
        # and [150-179], with lower saturation constraint to handle bright glowing lights
        lower_red1 = np.array([0, 50, 150])
        upper_red1 = np.array([30, 255, 255])
        lower_red2 = np.array([150, 50, 150])
        upper_red2 = np.array([179, 255, 255])
        
        # Green: H in [40-90], S > 80, V > 80
        lower_green = np.array([40, 80, 80])
        upper_green = np.array([90, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        
        # Determine the strongest color
        max_pixels = max(red_pixels, green_pixels)
        
        # Default to RED if no prominent color is found (or if washed out)
        raw_state = SignalState.RED
        
        if max_pixels > 5:
            if max_pixels == green_pixels:
                raw_state = SignalState.GREEN
            else:
                raw_state = SignalState.RED
                
        self.current_state = self._get_dominant_state(raw_state)
        return self.current_state


class SimulatedSignalDetector(TrafficSignalDetector):
    """
    Overrides the detect method to simply toggle between GREEN and RED 
    every N seconds based on the system clock.
    """
    def __init__(self, roi_polygon: List[Tuple[int, int]], toggle_interval_sec: float = 10.0):
        super().__init__(roi_polygon)
        self.toggle_interval_sec = toggle_interval_sec
        # Determine initial state by offset
        self.start_time = time.time()

    def detect(self, frame: np.ndarray) -> SignalState:
        elapsed = time.time() - self.start_time
        # Integer division by interval to toggle
        cycle_count = int(elapsed // self.toggle_interval_sec)
        
        # Even cycle = GREEN, Odd cycle = RED
        if cycle_count % 2 == 0:
            return SignalState.GREEN
        else:
            return SignalState.RED
