import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from .calibration import transform_points

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

   
    model = YOLO(model_path)

   
    video_info = sv.VideoInfo.from_video_path(input_video)
    frame_generator = sv.get_video_frames_generator(input_video)

    # This initialises the multi-object tracker
    tracker = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=conf_threshold
    )

    
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)

  
     
    mask_annot = sv.MaskAnnotator(
    opacity=0.4,      
    color_lookup=sv.ColorLookup.TRACK

)
    mask_outline_annot = sv.MaskAnnotator(
    opacity=0.0,   
   
    color_lookup=sv.ColorLookup.TRACK
)


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

    # Defines the box for detections in the road
    zone = sv.PolygonZone(polygon=road_polygon)

   
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        video_info.fps,
        video_info.resolution_wh
    )

  
    vehicle_ids = set()

  
    for frame in tqdm(frame_generator, total=video_info.total_frames):

     
        results = model(
            frame,
            imgsz=model_resolution,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )[0]

    
        detections = sv.Detections.from_ultralytics(results)

        # This filter and removes detections to include only vehicle classes of interest
        detections = detections[np.isin(detections.class_id, vehicle_classes)]
        detections = detections[zone.trigger(detections)]

    
        detections = tracker.update_with_detections(detections)

        # If no valid detections are present, write the original frame
        if len(detections) == 0:
            writer.write(frame)
            continue

    
        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # Convert image-space points into real-world coordinates
       
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

      
        annotated = frame.copy()

        # This will draw the road area the used for calibration
        annotated = sv.draw_polygon(
            annotated,
            road_polygon,
            color=sv.Color.RED,
            thickness=2
        )

        # This draws the bounding boxes, trajectories, and labels
        
        annotated = mask_annot.annotate(annotated, detections)
        annotated = mask_outline_annot.annotate(annotated, detections)

        annotated = trace_annot.annotate(annotated, detections)
        annotated = label_annot.annotate(annotated, detections, labels)

        # This displays the total number of unique vehicles detected
        cv2.putText(
            annotated,
            f"Vehicles counted: {len(vehicle_ids)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

         
        writer.write(annotated)

   
    writer.release()
