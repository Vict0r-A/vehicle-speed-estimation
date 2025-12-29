import numpy as np
from collections import defaultdict, deque


class SpeedEstimator:
    """
    Estimates vehicle speed in kilometres per hour using tracked positions
    projected into real-world coordinates.
    """

    def __init__(self, fps):
  
        self.fps = fps

  
        self.coordinates = defaultdict(lambda: deque(maxlen=fps))

    def calculate_speed(self, track_id, world_point):
        """
        This functions purpose is to calculate the current speed of a tracked vehicle.
        """
 
        coords = self.coordinates[track_id]
        coords.append(world_point)
        if len(coords) < 2:
            return None

        distance_m = np.linalg.norm(coords[-1] - coords[0])

        time_s = len(coords) / self.fps


        speed_mps = distance_m / time_s
        return speed_mps * 3.6
