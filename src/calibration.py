import cv2
import numpy as np


def compute_homography(road_polygon, road_width_m, road_length_m):
    """
    This function computes a homography that maps image-space road coordinates
    to a real-world ground plane measured in metres.
    """

     
    world_box = np.array([
        [0, 0],
        [road_width_m - 1, 0],
        [road_width_m - 1, road_length_m - 1],
        [0, road_length_m - 1]
    ], dtype=np.float32)

     
    homography = cv2.getPerspectiveTransform(
        road_polygon.astype(np.float32),
        world_box
    )

    return homography


def transform_points(points, homography):
    """
    This function transforms the image points into real-world coordinates using homography.
    """

    
    if points.size == 0:
        return points

    pts = points.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, homography)
    return warped.reshape(-1, 2)
