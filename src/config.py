import numpy as np

"""
This module is to set configuration settings for the project.

"""


INPUT_VIDEO = "vehicles.mp4"
OUTPUT_VIDEO = "output_video/vehicles_output.mp4"
MODEL = "yolov8m-seg.pt"
MODEL_RESOLUTION = 1280
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# COCO class IDs corresponding to vehicles - car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]   

#These are the dimesnions for the 'box'
ROAD_WIDTH_M = 25
ROAD_LENGTH_M = 250

# This array defines the road region via coordinates provided by roboflow
ROAD_POLYGON = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])
