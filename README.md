# Vehicle Damage Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111111)
![Status](https://img.shields.io/badge/Status-Ready-success)

Simple Streamlit app for vehicle damage detection using a trained YOLOv8 model (`best.pt`).

## What it does

1. Upload a car image.
2. Detect damage regions with bounding boxes.
3. Show for each detection:
   - Damage Type
   - Severity (minor, moderate, severe)
   - Estimated Car Part (front, rear, door/side, wheel/bumper)

## Input image types

- `.jpg`
- `.jpeg`
- `.png`

Use clear car photos where damaged areas are visible.

## Installation

1. Open terminal in this folder.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Result you will see

- Uploaded image preview
- Image with YOLO bounding boxes and labels
- A concise summary table in this format:

```text
Damage Type | Severity | Car Part
```

If no damage is found, the app shows: **No damage detected**.

## Project files

- `app.py` - Streamlit application
- `best.pt` - Trained YOLOv8 weights
- `requirements.txt` - Python dependencies
