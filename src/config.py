import numpy as np

"""
Configuration settings for the vehicle speed estimation project.

"""

# Input and output video paths
INPUT_VIDEO = "vehicles.mp4"
OUTPUT_VIDEO = "output_video/vehicles_output.mp4"

# YOLOv8 model configuration
MODEL = "yolov8m.pt"
MODEL_RESOLUTION = 1280
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# COCO class IDs corresponding to vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Ground-truth road dimensions in metres
# These values were confirmed by Roboflow for the supplied video
ROAD_WIDTH_M = 25
ROAD_LENGTH_M = 250

# Polygon defining the road region in image coordinates
# Used for region filtering and perspective calibration
ROAD_POLYGON = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])
