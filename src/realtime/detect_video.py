"""
Playing Card Detection on Videos
Detects and tracks playing cards in video files or webcam feed
python -m src.realtime.detect_video webcam
"""

from ultralytics import YOLO
import cv2
import sys

def detect_video(model_path, video_path, conf_threshold=0.6, save=True, show=False):
    """
    Run card detection on a video file.
    
    Args:
        model_path: Path to trained YOLO model
        video_path: Path to video file
        conf_threshold: Confidence threshold for detections
        save: Whether to save the output video
        show: Whether to display video in real-time
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run prediction on video
    print(f"Processing video: {video_path}")
    print(f"Confidence threshold: {conf_threshold}")
    
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=save,
        show=show,
        stream=True,  # Process video frame by frame
        verbose=False,  # Reduce output clutter
        project='examples/result',
        name='video',
        exist_ok=True  # Allow overwriting/reusing the same directory
    )
    
    # Process results frame by frame
    frame_count = 0
    total_detections = 0
    
    for result in results:
        frame_count += 1
        boxes = result.boxes
        num_detections = len(boxes)
        total_detections += num_detections
        
        if num_detections > 0:
            print(f"Frame {frame_count}: {num_detections} cards detected")
            
            # Print card details
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                print(f"  - {class_name}: {confidence:.2f}")
    
    print(f"\nProcessing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections/frame_count:.2f}")
    
    if save:
        print(f"Output video saved to: examples/result/video/")


def detect_webcam(model_path, conf_threshold=0.25):
    """
    Run real-time card detection on webcam feed.
    
    Args:
        model_path: Path to trained YOLO model
        conf_threshold: Confidence threshold for detections
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print("Starting webcam detection...")
    print("Press 'q' to quit")
    
    # Open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Display frame
        cv2.imshow('Card Detection', annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped")


def batch_process_videos(model_path, video_folder, conf_threshold=0.25):
    """
    Process multiple videos in a folder.
    
    Args:
        model_path: Path to trained YOLO model
        video_folder: Path to folder containing videos
        conf_threshold: Confidence threshold for detections
    """
    import os
    from pathlib import Path
    
    model = YOLO(model_path)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_folder).glob(f'*{ext}'))
    
    print(f"Found {len(video_files)} videos to process")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        
        results = model.predict(
            source=str(video_path),
            conf=conf_threshold,
            save=True,
            stream=True,
            verbose=False,
            project='examples/result',
            name='video',
            exist_ok=True  # Allow overwriting/reusing the same directory
        )
        
        # Process results
        for _ in results:
            pass  # Just iterate through to process
        
        print(f"Completed: {video_path.name}")
    
    print(f"\nAll videos processed! Results saved to: examples/result/video/")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "runs/detect/train_corners_v1/weights/best.pt"
    CONFIDENCE = 0.6
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Video file:  py detect_video.py path/to/video.mp4")
        print("  Webcam:      py detect_video.py webcam")
        print("  Batch:       py detect_video.py batch path/to/video/folder/")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode.lower() == "webcam":
        # Webcam mode
        detect_webcam(MODEL_PATH, CONFIDENCE)
    
    elif mode.lower() == "batch":
        # Batch processing mode
        if len(sys.argv) < 3:
            print("Error: Please provide folder path for batch processing")
            sys.exit(1)
        folder_path = sys.argv[2]
        batch_process_videos(MODEL_PATH, folder_path, CONFIDENCE)
    
    else:
        # Video file mode
        video_path = sys.argv[1]
        
        # Optional: custom confidence threshold
        if len(sys.argv) > 2:
            CONFIDENCE = float(sys.argv[2])
        
        detect_video(MODEL_PATH, video_path, CONFIDENCE, save=True, show=False)
