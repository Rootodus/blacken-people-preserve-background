import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import subprocess
import random
import string

def random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def save_image(image, output_path):
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"Image saved at {output_path}")
    else:
        print(f"Failed to save image at {output_path}")

def black_out_image(input_image_path, output_image_path, model_path='yolov8n-seg.pt'):
    model = YOLO(model_path)
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not load image {input_image_path}")
        return

    results = model(image)[0]

    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for i, cls_id in enumerate(results.boxes.cls):
        if int(cls_id) == 0:  # person class
            mask_small = results.masks.data[i].cpu().numpy() > 0.5
            mask = cv2.resize(mask_small.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_mask = cv2.bitwise_or(combined_mask, mask * 255)

    if combined_mask.sum() == 0:
        print("No person detected. Saving original image.")
        save_image(image, output_image_path)
        return

    # Dilate mask to cover more area (approximate person more generously)
    kernel = np.ones((15, 15), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=10)
    
    output = image.copy()
    output[combined_mask == 255] = (0, 0, 0)
    
    save_image(output, output_image_path)

def black_out_video(input_video_path, output_video_path, model_path='yolov8n-seg.pt', detect_every_n_frames=5, dilation_iterations=10):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {frame_count} frames")

    last_mask = np.zeros((h, w), dtype=np.uint8)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute scene change
        scene_change_threshold = 20.0  # tune this
        if frame_idx == 0:
            prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        run_yolo = False

        if frame_idx % detect_every_n_frames == 0:
            run_yolo = True
        else:
            # compare current frame to previous YOLO frame
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(current_gray, prev_frame_gray)
            mean_diff = np.mean(diff)
            if mean_diff > scene_change_threshold:
                run_yolo = True

        if run_yolo:
            results = model(frame)[0]
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for i, cls_id in enumerate(results.boxes.cls):
                if int(cls_id) == 0:  # person
                    mask_small = results.masks.data[i].cpu().numpy() > 0.5
                    mask = cv2.resize(mask_small.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    combined_mask = cv2.bitwise_or(combined_mask, mask * 255)

            # Dilate mask to cover more area (approximate person more generously)
            kernel = np.ones((15, 15), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=dilation_iterations)

            last_mask = combined_mask.copy()
            prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use last computed mask
        output = frame.copy()
        output[last_mask == 255] = (0, 0, 0)

        out.write(output)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video saved at {output_video_path}")

def add_audio(video_path, original_path, output_path):
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', original_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_path
    ]
    subprocess.run(cmd)
    print(f"Final video with audio saved at {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python blacken_content.py <input_image_or_video>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: file '{input_path}' does not exist.")
        sys.exit(1)

    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    if ext in ('.jpg', '.jpeg', '.png'):
        output_path = f"blackened_image{ext}"
        black_out_image(input_path, output_path)

    elif ext in ('.mp4', '.mkv', '.mov'):
        gibberish = random_string()
        output_path = f"blackened_video_{gibberish}{ext}"
        black_out_video(input_path, output_path)
        final_output = f"blackened_video{ext}"
        add_audio(output_path, input_path, final_output)
        
        # Delete intermediate video to save space
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Deleted intermediate video: {output_path}")

    else:
        print("Unsupported file type.")
