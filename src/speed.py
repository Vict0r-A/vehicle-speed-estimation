import numpy as np
from collections import defaultdict, deque


class SpeedEstimator:
    """
    Estimates vehicle speed in kilometres per hour using tracked positions
    projected into real-world coordinates.
    """

    def __init__(self, fps):
  
        self.fps = fps

        # Store recent world-coordinate positions for each tracked vehicle
        self.coordinates = defaultdict(lambda: deque(maxlen=fps))

    def calculate_speed(self, track_id, world_point):
        """
        Calculate the current speed of a tracked vehicle.
        """

        # Append the latest world-space position for this vehicle
        coords = self.coordinates[track_id]
        coords.append(world_point)

        # Require at least two points to estimate motion
        if len(coords) < 2:
            return None

        # Compute distance travelled in metres over the stored time window
        distance_m = np.linalg.norm(coords[-1] - coords[0])

        # Convert number of frames into elapsed time in seconds
        time_s = len(coords) / self.fps

        # Convert metres per second to kilometres per hour
        speed_mps = distance_m / time_s
        return speed_mps * 3.6
