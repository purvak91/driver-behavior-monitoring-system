"""
Zebra Crossing Detector — Detects the zebra crossing (white stripe markings)
in the frame and returns a stop-line Y coordinate derived from it.

Strategy:
  1. Focus on the bottom 45% of the frame where the crossing appears.
  2. Convert to grayscale, threshold for bright white marks.
  3. Use morphological filtering to isolate horizontal stripe-like contours.
  4. Cluster the detected stripes and compute the stop-line as their top edge.
  5. Cache the result so we only recompute every N frames (the crossing doesn't move).
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class ZebraCrossingDetector:
    def __init__(self, recalc_interval: int = 150):
        """
        Args:
            recalc_interval: Re-detect zebra crossing every N frames.
        """
        self.recalc_interval = recalc_interval
        self._frame_counter = 0
        self._cached_stop_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self._cached_zebra_boxes: List[Tuple[int, int, int, int]] = []

    def detect(self, frame) -> Tuple[Optional[Tuple[Tuple[int, int], Tuple[int, int]]], List]:
        """
        Detect zebra crossing in the given frame.
        
        Returns:
            (stop_line, zebra_boxes) where stop_line is ((x1,y1),(x2,y2))
            or (None, []) if no zebra detected.
        """
        self._frame_counter += 1

        # Use cache if available and not time to recalculate
        if self._cached_stop_line is not None and self._frame_counter % self.recalc_interval != 0:
            return self._cached_stop_line, self._cached_zebra_boxes

        height, width = frame.shape[:2]

        # --- ROI: bottom 45% of the frame ---
        roi_top = int(height * 0.55)
        roi = frame[roi_top:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # --- Threshold for bright white road markings ---
        _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

        # Morphological operations: close small gaps, then open to remove noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        zebra_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # Zebra stripes are: moderately large, wider-than-tall rectangles
            # Filter out huge blobs (sky, walls) and tiny noise
            if 800 < area < 50000 and aspect_ratio > 1.5 and w > 40 and h < 80:
                zebra_candidates.append((x, y + roi_top, w, h))

        if len(zebra_candidates) >= 3:
            # Sort by Y (top-most first)
            zebra_candidates.sort(key=lambda b: b[1])

            # The stop line should be at the TOP edge of the zebra crossing
            # Use 25th percentile Y to be robust against outliers
            y_values = [b[1] for b in zebra_candidates]
            stop_y = int(np.percentile(y_values, 25))

            stop_line = ((0, stop_y), (width, stop_y))
            self._cached_stop_line = stop_line
            self._cached_zebra_boxes = zebra_candidates
            return stop_line, zebra_candidates

        # If detection failed but we have a previous cache, keep using it
        if self._cached_stop_line is not None:
            return self._cached_stop_line, self._cached_zebra_boxes

        # Ultimate fallback: place stop line at 70% frame height
        fallback_y = int(height * 0.70)
        fallback_line = ((0, fallback_y), (width, fallback_y))
        return fallback_line, []

    def get_stop_line(self, frame) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convenience method: returns only the stop line."""
        line, _ = self.detect(frame)
        return line
