import os
import sys
from ultralytics import YOLO

def forFrame(frame_number, output_objects_array, output_objects_count):
    pass

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python fire_net.py --video <video_path>")
        print("python fire_net.py --camera <camera_index>")
        print("python fire_net.py --image <image_path>")
        sys.exit(1)

    # load YOLO11 model trained on fire dataset
    model_path = "runs/detect/train3/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}")
        print("Please wait for training to complete or use the pre-trained model")
        model_path = "yolo11n.pt" 
    
    model = YOLO(model_path)

    if sys.argv[1] == "--image":
        image_path = sys.argv[2]

        if not os.path.exists(image_path):
            print("Image not found:", image_path)
            sys.exit(1)

        results = model(image_path, conf=0.3)
        
        fire_found = False
        for result in results:
            result.save(filename="fire_image_output.jpg")
            for box in result.boxes:
                cls_id = int(box.cls[0]) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0]) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                print(f"Detected {cls_name} with confidence {conf:.2f}")
                if cls_name.lower() == "fire":
                    fire_found = True

        if fire_found:
            print("Fire detected in image. Output saved to fire_image_output.jpg")
        else:
            print("No fire detected in image. Output saved to fire_image_output.jpg")

        sys.exit(0)

    if sys.argv[1] == "--video":
        video_path = sys.argv[2]

        if not os.path.exists(video_path):
            print("Video not found:", video_path)
            sys.exit(1)

        # for video detection
        results = model(video_path, conf=0.3, save=True, project=".", name="fire_output")

    elif sys.argv[1] == "--camera":
        camera_index = int(sys.argv[2])

        # for camera, we need to process frames
        import cv2
        cap = cv2.VideoCapture(camera_index)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, conf=0.3)

            
        cap.release()

    else:
        print("Invalid argument")
        sys.exit(1)

if __name__ == "__main__":
    main()
