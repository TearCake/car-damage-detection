from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parent / "best.pt"
CONF_THRESHOLD = 0.2


def classify_severity(damage_type: str) -> str:
    severe_labels = {"crack", "glass_damage"}
    moderate_labels = {"dent", "hole", "part_damage"}
    minor_labels = {"paint_damage"}

    if damage_type in severe_labels:
        return "severe"
    if damage_type in moderate_labels:
        return "moderate"
    if damage_type in minor_labels:
        return "minor"
    return "moderate"


def estimate_car_part(x1: float, y1: float, x2: float, y2: float, image_shape: tuple[int, int]) -> str:
    height, width = image_shape
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    if center_x < width * 0.33:
        horizontal_zone = "front"
    elif center_x > width * 0.66:
        horizontal_zone = "rear"
    else:
        horizontal_zone = "middle"

    if center_y > height * 0.70:
        return "wheel / bumper"
    if horizontal_zone == "front":
        return "front"
    if horizontal_zone == "rear":
        return "rear"
    return "door / side"


@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    return YOLO(str(model_path))


def run_prediction(model: YOLO, image_rgb: np.ndarray) -> tuple[np.ndarray, list[dict[str, str]]]:
    results = model.predict(source=image_rgb, conf=CONF_THRESHOLD, verbose=False)
    if not results:
        return image_rgb, []

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return image_rgb, []

    plotted_bgr = result.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

    height, width = result.orig_shape
    detections: list[dict[str, str]] = []

    for box in boxes:
        class_id = int(box.cls[0].item())
        if isinstance(model.names, dict):
            damage_type = model.names.get(class_id, str(class_id))
        else:
            damage_type = model.names[class_id] if class_id < len(model.names) else str(class_id)

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        severity = classify_severity(damage_type)
        car_part = estimate_car_part(x1, y1, x2, y2, (height, width))

        detections.append(
            {
                "damage_type": damage_type,
                "severity": severity,
                "car_part": car_part,
            }
        )

    return plotted_rgb, detections


def main() -> None:
    st.set_page_config(page_title="Vehicle Damage Detection System", layout="centered")
    st.title("Vehicle Damage Detection System")
    st.caption("Upload an image to detect car damages with YOLOv8")
    st.divider()

    if not MODEL_PATH.exists():
        st.error("Model file best.pt was not found in the app directory.")
        st.stop()

    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a car image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Please upload an image file (.jpg, .jpeg, .png) to start detection.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image_rgb, use_container_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Running damage detection..."):
            try:
                plotted_image, detections = run_prediction(model, image_rgb)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        st.divider()
        st.subheader("Detected Damage (Bounding Boxes)")
        st.image(plotted_image, use_container_width=True)

        st.subheader("Damage Summary")
        if not detections:
            st.warning("No damage detected")
            return

        st.markdown("Damage Type | Severity | Car Part")
        st.markdown("---|---|---")
        for item in detections:
            st.markdown(f"{item['damage_type']} | {item['severity']} | {item['car_part']}")


if __name__ == "__main__":
    main()
