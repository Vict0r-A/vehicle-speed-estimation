import cv2
import numpy as np


def compute_homography(road_polygon, road_width_m, road_length_m):
    """
    Compute a homography that maps image-space road coordinates
    to a real-world ground plane measured in metres.
    """

    # Define a rectangular target plane in real-world coordinates
    # The rectangle represents the road surface measured in metres
    world_box = np.array([
        [0, 0],
        [road_width_m - 1, 0],
        [road_width_m - 1, road_length_m - 1],
        [0, road_length_m - 1]
    ], dtype=np.float32)

    # Compute the perspective transform from image space to world space
    homography = cv2.getPerspectiveTransform(
        road_polygon.astype(np.float32),
        world_box
    )

    return homography


def transform_points(points, homography):
    """
    Transform a set of image-space points into real-world coordinates
    using a precomputed homography.
    """

    # Handle the case where no points are provided
    if points.size == 0:
        return points

    # Reshape points to match OpenCV's expected input format
    pts = points.reshape(-1, 1, 2).astype(np.float32)

    # Apply the perspective transformation
    warped = cv2.perspectiveTransform(pts, homography)

    # Return points in a flattened (N, 2) format
    return warped.reshape(-1, 2)
