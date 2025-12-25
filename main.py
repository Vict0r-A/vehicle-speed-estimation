from supervision.assets import VideoAssets, download_assets
from src.config import *
from src.calibration import compute_homography
from src.speed import SpeedEstimator
from src.video import run
import os


def main():
    """
    Entry point for the vehicle speed estimation project.

    """

    # Download the demo video asset if it is not already present
    download_assets(VideoAssets.VEHICLES)

    # Ensure the output directory exists before writing the result video
    os.makedirs("output_video", exist_ok=True)

    # Compute the homography using confirmed real-world road dimensions
    homography = compute_homography(
        ROAD_POLYGON,
        ROAD_WIDTH_M,
        ROAD_LENGTH_M
    )

    # Initialise the speed estimator using the video frame rate
    # The FPS value is used to convert frame counts into elapsed time
    speed_estimator = SpeedEstimator(fps=30)

    # Run the full detection, tracking, and speed estimation pipeline
    run(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        MODEL,
        MODEL_RESOLUTION,
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        VEHICLE_CLASSES,
        ROAD_POLYGON,
        homography,
        speed_estimator
    )


if __name__ == "__main__":
    main()

