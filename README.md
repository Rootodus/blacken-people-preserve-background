# Blacken People

This Python program detects people in an image or video and blackens them using [YOLOv8 segmentation](https://docs.ultralytics.com).\
It supports both images (`.jpg`, `.jpeg`, `.png`) and videos (`.mp4`, `.mkv`, `.mov`), and keeps the original audio track for videos.

For videos, the mask is also expanded (dilated) slightly to ensure full coverage of people, and the same applies to images.

---

## Features

✅ Detects people in images and videos using YOLOv8 segmentation.\
✅ Blacks out (masks) detected people while leaving the rest intact.\
✅ Retains original audio in videos.\
✅ Supports `.jpg`, `.jpeg`, `.png`, `.mp4`, `.mkv`, `.mov`.\
✅ Intermediate files are cleaned up automatically.

---

## Requirements

- Python 3.7 or higher
- [YOLOv8 weights file](https://github.com/ultralytics/ultralytics) (`yolov8n-seg.pt`) in the same folder as the script.

---

## Installation & Usage

### 1️⃣ Download and install Python

- Download Python from [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Make sure to check the option **Add Python to PATH** during installation.

---

### 2️⃣ Prepare your folder

- Create a new folder.
- Place `blacken_people.py` in this folder.
- Put the YOLO weights file `yolov8n-seg.pt` in the same folder.
- Place your input image or video in the folder, too.
- Open a **Command Prompt** or **Terminal** in this folder.

---

### 3️⃣ Install dependencies

Run the following command to install all required Python packages:

```bash
pip install opencv-python numpy ultralytics moviepy
```

---

### 4️⃣ Run the program

For an **image**:

```bash
python blacken_people.py input_image.jpg
```

For a **video**:

```bash
python blacken_people.py input_video.mp4
```

Output files will be saved in the same folder:

- Images: `blackened_image.jpg`
- Videos: `blackened_video.mp4`

---

## Notes

- If no people are detected, the original image/video is saved unchanged.
- For videos, a temporary intermediate file is created and deleted automatically.
- To adjust how much the blackened area extends beyond the person (dilation), you can edit the `dilation_iterations` value in the code.
