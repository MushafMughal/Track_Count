import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import math

# Configurations
VIDEO_PATH =  "C:/Users/musha/OneDrive/Desktop/Track_Count/YOLOv9_DeepSORT/yolov9/data/Clip2.mp4" # Path to input video or webcam index (0)
OUTPUT_PATH = 'C:/Users/musha/OneDrive/Desktop/Track_Count/YOLOv9_DeepSORT/yolov9/outputs'  # Path to save the processed video
YOLO_WEIGHTS = 'C:/Users/musha/Downloads/Yolo9-DeepSort-main/Yolo9-DeepSort-main/yolov9-c.pt'  # Path to YOLO model weights
COCO_CLASSES_PATH = 'C:/Users/musha/OneDrive/Desktop/Track_Count/YOLOv9_DeepSORT/configs/coco.names'  # Path to class labels
CONFIDENCE_THRESHOLD = 0.45  # Confidence threshold for detections
BLUR_PEOPLE = False  # Set True to blur detected people (class ID 0)

def main():
    # Initialize video input
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return
    
    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    
    # Load YOLO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights=YOLO_WEIGHTS, device=device, fuse=True)
    model = AutoShape(model)

    # Load COCO class labels
    with open(COCO_CLASSES_PATH, 'r') as f:
        class_names = f.read().strip().split("\n")
    
    # Generate random color for "person" class (ID 0)
    person_color = tuple(np.random.randint(0, 255, 3).tolist())

    # Initialize DeepSort tracker
    tracker = DeepSort(
        max_iou_distance=0.8,
        max_age=10,
        n_init=3,
        nms_max_overlap=0.5,
        max_cosine_distance=0.35,
        nn_budget=100)
    
    # Define the coordinates for the line
    LINE_START = (450, 275)
    LINE_END = (600, 150) # (600,100) for optimal performance
    LINE_COLOR = (0, 0, 255)  # Red color
    LINE_THICKNESS = 2  # Thickness of the line

    # Define frame skip
    frame_skip = 6

    # Initialize counter and store crossed IDs
    people_count = 0
    crossed_ids = set()
    
    frame_no = 0
    while True:
        ret, frame = cap.read()        
        if not ret:
            break
        
        frame_no+=1
        if frame_no % frame_skip == 0:
            
            
            # Draw the line on the frame
            cv2.line(frame, LINE_START, LINE_END, LINE_COLOR, LINE_THICKNESS)
            
            # Run YOLO model on the frame
            results = model(frame)
            detections = results.xyxy[0]  # YOLO detections as (x1, y1, x2, y2, conf, class_id)

            detect = []
            for det in detections:
                x1, y1, x2, y2, confidence, class_id = map(float, det)
                class_id = int(class_id)
                
                # Filter detections for "person" class (ID 0) and confidence threshold
                if class_id == 0 and confidence >= CONFIDENCE_THRESHOLD:
                    detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
            
            # Update tracks with detections
            tracks = tracker.update_tracks(detect, frame=frame)
            
            for track in tracks:
                
                if int(track.track_id) >= 3:
                    print("mean",track.track_id)
                    print(track.mean[4],track.mean[5])
                if not track.is_confirmed():
                    continue
                
                # Extract tracking details
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
                label = f"ID {track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 2)
                
                if track_id not in crossed_ids:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    # Calculate A, B, C for the line equation
                    A = LINE_END[1] - LINE_START[1]
                    B = LINE_START[0] - LINE_END[0]
                    C = LINE_END[0] * LINE_START[1] - LINE_START[0] * LINE_END[1]

                    # Compute the distance
                    distance = (A * center_x + B * center_y + C) / math.sqrt(A**2 + B**2)
                    if distance > 0 and abs(distance) < 50 and track.mean[4] > 0.0:
                        
                        crossed_ids.add(track_id)
                        people_count += 1
                
            
                # Apply blur if enabled
                if BLUR_PEOPLE:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 30)
            
            # Display the count on the frame
            cv2.putText(frame, f"People Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show and save the frame
            cv2.imshow('Person Tracking', frame)
            writer.write(frame)
            
            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
