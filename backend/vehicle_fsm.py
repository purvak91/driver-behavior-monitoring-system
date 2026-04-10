import os
import cv2
import time
from enum import Enum
from typing import Dict, Tuple, Optional
from signal_detector import SignalState

class VehicleState(Enum):
    MOVING = "MOVING"
    STOPPED = "STOPPED"
    VIOLATED = "VIOLATED"
    CLEARED = "CLEARED"

class VehicleFSM:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.state = VehicleState.MOVING
        
        # Keep recent (x, y) centroids to calculate movement and line intersection
        self.centroids = []
        self.has_crossed = False
        
    def _ccw(self, A, B, C):
        """Returns True if points A, B, C are in counter-clockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
    def _segments_intersect(self, A, B, C, D):
        """Returns True if line segment AB intersects with line segment CD."""
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)
        
    def _get_side(self, p1, p2, point):
        """
        Computes side of the line p1-p2. 
        side(point) = sign((x2 - x1)*(y - y1) - (y2 - y1)*(x - x1))
        """
        x1, y1 = p1
        x2, y2 = p2
        x, y = point
        val = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
        if val > 0: return 1
        elif val < 0: return -1
        return 0

    def update(self, centroid_x: int, centroid_y: int, signal_state: SignalState, stop_line: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """
        Updates the FSM and returns True if a NEW violation occurred in this frame.
        """
        curr_point = (centroid_x, centroid_y)
        self.centroids.append(curr_point)
        
        # Keep only the last 10 frames
        if len(self.centroids) > 10:
            self.centroids.pop(0)

        p1, p2 = stop_line
        just_violated = False
        just_crossed = False
        
        # Check if crossed the line recently using side-of-line (cross product)
        # This treats the user's two points as defining an INFINITE line,
        # so vehicles are detected even if they cross beyond the endpoints.
        if len(self.centroids) >= 2:
            prev_point = self.centroids[-2]
            
            side_prev = self._get_side(p1, p2, prev_point)
            side_curr = self._get_side(p1, p2, curr_point)
            
            # A crossing occurs when the vehicle moves from one side to the other
            if side_prev != 0 and side_curr != 0 and side_prev != side_curr:
                just_crossed = True
                self.has_crossed = True

        # Check for red light violation
        if self.state in [VehicleState.MOVING, VehicleState.STOPPED]:
            # 1. Running the red light (crossed while red)
            if just_crossed and signal_state == SignalState.RED:
                self.state = VehicleState.VIOLATED
                just_violated = True
            # 2. Stopped completely, but past the line while light is red (if it wasn't already recorded)
            # To handle this fairly without the old y_thresh, if we crossed it while red, it will be detected above.
            # No need for arbitrary y_thresh logic here now.

        # Calculate Y movement over the available frame window
        y_movement_recent_10 = abs(self.centroids[-1][1] - self.centroids[0][1]) if len(self.centroids) == 10 else 100
        y_movement_recent_3 = abs(self.centroids[-1][1] - self.centroids[-3][1]) if len(self.centroids) >= 3 else 100

        # Handle State Transitions
        if self.state == VehicleState.MOVING:
            if y_movement_recent_10 < 5 and len(self.centroids) == 10:
                self.state = VehicleState.STOPPED
        elif self.state == VehicleState.STOPPED:
            if y_movement_recent_3 > 15:
                self.state = VehicleState.MOVING
        elif self.state == VehicleState.VIOLATED:
            if signal_state == SignalState.GREEN and y_movement_recent_3 > 15:
                self.state = VehicleState.CLEARED

        return just_violated


class FSMManager:
    def __init__(self, stop_line: Tuple[Tuple[int, int], Tuple[int, int]]):
        self.vehicles: Dict[int, VehicleFSM] = {}
        self.stop_line = stop_line
        
        # Ensure violations directory exists
        self.violations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "violations")
        os.makedirs(self.violations_dir, exist_ok=True)

    def set_stop_line(self, new_line: Tuple[Tuple[int, int], Tuple[int, int]]):
        self.stop_line = new_line

    def unregister_stale_vehicles(self, active_track_ids: list):
        """Removes vehicles that are no longer tracked."""
        stale_keys = [k for k in self.vehicles.keys() if k not in active_track_ids]
        for k in stale_keys:
            del self.vehicles[k]

    def update(self, frames_info: dict, frame, signal_state: SignalState):
        """
        frames_info: dict { track_id: {'plate': 'MH12...', 'bbox': (x1,y1,x2,y2)} }
        """
        active_violation_events = []
        
        for track_id, info in frames_info.items():
            if track_id not in self.vehicles:
                self.vehicles[track_id] = VehicleFSM(track_id)
                
            x1_bb, y1_bb, x2_bb, y2_bb = info['bbox']
            centroid_x = (x1_bb + x2_bb) // 2
            centroid_y = (y1_bb + y2_bb) // 2
            
            fsm = self.vehicles[track_id]
            is_new_violation = fsm.update(centroid_x, centroid_y, signal_state, self.stop_line)
            
            if is_new_violation:
                # Save snapshot
                timestamp = int(time.time())
                plate = info.get('plate', f"UNK-{track_id}")
                filename = f"red_violation_{plate}_{timestamp}.jpg"
                filepath = os.path.join(self.violations_dir, filename)
                
                # Draw stop-line and violation tag for visual proof
                shot = frame.copy()
                (sx1, sy1), (sx2, sy2) = self.stop_line
                cv2.line(shot, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
                cv2.putText(shot, "RED LIGHT VIOLATION", (x1_bb, y1_bb - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(filepath, shot)
                
                active_violation_events.append({
                    "track_id": track_id,
                    "plate": plate,
                    "state": fsm.state.value,
                    "timestamp": timestamp,
                    "frame_path": filepath
                })
                
        self.unregister_stale_vehicles(list(frames_info.keys()))
        return active_violation_events
