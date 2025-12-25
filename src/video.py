import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm


def run(
    input_video,
    output_video,
    model_path,
    model_resolution,
    conf_threshold,
    iou_threshold,
    vehicle_classes,
    road_polygon,
    homography,
    speed_estimator
):
    """
    Runs the full vehicle detection, tracking, and speed estimation pipeline
    on a single input video and writes an annotated output video.
    """

    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Extract video metadata and create a frame generator
    video_info = sv.VideoInfo.from_video_path(input_video)
    frame_generator = sv.get_video_frames_generator(input_video)

    # Initialise the multi-object tracker
    tracker = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=conf_threshold
    )

    # Compute annotation scaling based on video resolution
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)

    # Initialise annotation utilities
    bbox_annot = sv.BoxAnnotator(thickness=thickness)
    label_annot = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annot = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER
    )

    # Define a polygon zone to restrict detections to the road area
    zone = sv.PolygonZone(polygon=road_polygon)

    # Set up video writer for the output file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        video_info.fps,
        video_info.resolution_wh
    )

    # Track unique vehicle IDs for counting
    vehicle_ids = set()

    # Process the video frame by frame
    for frame in tqdm(frame_generator, total=video_info.total_frames):

        # Run object detection on the current frame
        results = model(
            frame,
            imgsz=model_resolution,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )[0]

        # Convert model outputs into a Supervision Detections object
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections to include only vehicle classes of interest
        detections = detections[np.isin(detections.class_id, vehicle_classes)]

        # Remove detections outside the defined road polygon
        detections = detections[zone.trigger(detections)]

        # Update tracker to maintain consistent IDs across frames
        detections = tracker.update_with_detections(detections)

        # If no valid detections are present, write the original frame
        if len(detections) == 0:
            writer.write(frame)
            continue

        # Extract bottom-centre anchor points from bounding boxes
        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # Convert image-space points into real-world coordinates
        from .calibration import transform_points
        world_points = transform_points(points, homography)

        # Generate labels containing tracking ID and estimated speed
        labels = []
        for track_id, world_pt in zip(detections.tracker_id, world_points):
            speed = speed_estimator.calculate_speed(track_id, world_pt)
            vehicle_ids.add(track_id)

            if speed is None:
                labels.append(f"ID {track_id}")
            else:
                labels.append(f"ID {track_id} | {int(speed)} km/h")

        # Create an annotated copy of the current frame
        annotated = frame.copy()

        # Draw the road polygon used for calibration
        annotated = sv.draw_polygon(
            annotated,
            road_polygon,
            color=sv.Color.RED,
            thickness=2
        )

        # Draw bounding boxes, trajectories, and labels
        annotated = bbox_annot.annotate(annotated, detections)
        annotated = trace_annot.annotate(annotated, detections)
        annotated = label_annot.annotate(annotated, detections, labels)

        # Overlay the total number of unique vehicles detected
        cv2.putText(
            annotated,
            f"Vehicles counted: {len(vehicle_ids)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Write the annotated frame to the output video
        writer.write(annotated)

    # Release the video writer and finalise the output file
    writer.release()
