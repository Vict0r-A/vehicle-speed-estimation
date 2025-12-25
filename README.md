# Vehicle Speed Detection Project

**Author:** Victor Adams

## Overview
This project detects vehicles, tracks them, estimates vehicle speed and counts vehicles that enter and leave the video.

The project demonstrates applied computer vision techniques commonly used in **traffic analytics and intelligent transport systems**.

---

## Measurements

The input video (`vehicles.mp4`) is supplied by **Roboflow** as part of their public computer vision demo assets.

For this specific video, Roboflow has **confirmed the real-world dimensions** of the visible road section to be:

- **25 metres in width**
- **250 metres in length**

These verified measurements are used as **ground-truth dimensions** for the perspective transformation and are not estimated or assumed.

---

## Skills Demonstrated

### 1. Vehicle Object Detection
- Vehicles are detected using **YOLOv8 (Ultralytics)**.
- Only relevant vehicle classes are retained:
  - Car
  - Motorcycle
  - Bus
  - Truck

### 2. Region of Interest Filtering
- A polygonal region corresponding to the road surface is defined.
- Detections outside this region are discarded to reduce false positives.

### 3. Perspective Transformation
- A homography is computed between:
  - The road polygon in image space
  - A real-world rectangular plane of **250 × 25 metres**

This enables mapping from pixel coordinates to **metric world coordinates** on the ground plane.

### 4. Multi-Object Tracking
- **ByteTrack** is used to assign and maintain consistent IDs for vehicles across frames.
- Vehicle trajectories are stored over time.

### 5. Speed Estimation
- The bottom-centre of each bounding box is used as an approximation of the vehicle’s ground contact point.
- Vehicle positions are accumulated over time in world coordinates.
- Speed is calculated as distance travelled divided by elapsed time and reported in **km/h**.

### 6. Visualisation and Output
The output video includes:
- Bounding boxes and tracking IDs
- Vehicle trajectory lines
- Estimated speeds
- Total unique vehicle count
- Road region overlay

---

## Results

- Vehicles are consistently detected and tracked across the scene
- Estimated speeds fall within realistic urban traffic ranges
- Each vehicle is counted once using its unique tracking ID
- The project runs automatically and produces an annotated output video

<img width="1440" height="830" alt="Screenshot 2025-12-25 at 11 07 37" src="https://github.com/user-attachments/assets/434d05a8-906e-4292-a918-701c76d17412" />
---

## How to Run

You can run this project either **locally using VS Code and `main.py`** or **in Google Colab using a T4 GPU**.

---

## Option 1: Run Locally  (i.e VSCode)

### 1. Clone the repository
``` bash
git clone https://github.com/Vict0r-A/vehicle-speed-estimation.git
cd vehicle-speed-estimation
2. Create and activate a virtual environment (optional, but recommended)
Linux / macOS

python -m venv venv
source venv/bin/activate

Windows

python -m venv venv
venv\Scripts\activate
3. Install dependencies
 
pip install -r requirements.txt
4. Run the code
From the project root, run:

python -m main,py
The input video will be downloaded automatically, and the annotated output video will be saved to:

output_video/vehicles_output.mp4
```
### Option 2: Run in Google Colab (T4 GPU)
 
 # 1. Open the notebook in Google Colab
Upload Vehicle_Computer_Vision.ipynb to Google Colab
or open it directly from GitHub using the Open in Colab option

# 2. Select GPU runtime
In Colab:

Runtime → Change runtime type

Hardware accelerator → GPU

GPU type → T4

Save

# 3. Run the notebook

Run all cells from top to bottom


# 4. Output
The annotated output video will be generated as:

vehicles_output.mp4


