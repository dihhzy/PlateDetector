import streamlit as st
import cv2
import numpy as np
import re
import tempfile
import os
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models')

st.set_page_config(page_title="LPR Demo", layout="wide")
MODEL_DIR = '../models'

st.set_page_config(page_title="LPR Demo", layout="wide")


st.sidebar.title("⚙️ Settings")
input_type = st.sidebar.radio("Input Type", ["Image", "Video"])
flip_image = st.sidebar.checkbox("Flip Mirror Image (Horizontal)", value=False)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5) 


@st.cache_resource
def load_models():
    st.sidebar.text("Loading Models...")
    yolo_model = YOLO(os.path.join(MODEL_DIR, 'yolo11n.pt'))
    plate_model = YOLO(os.path.join(MODEL_DIR, 'plate_yolo11.pt'))
    
    ocr_model = LicensePlateRecognizer('global-plates-mobile-vit-v2-model', device='cuda')
    #ocr_model = LicensePlateRecognizer('global-plates-mobile-vit-v2-model', device='cpu')
    return yolo_model, plate_model, ocr_model

try:
    model, plate_model, ocr = load_models()
    st.sidebar.success("Models Loaded")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")

st.title("🚗 Car & License Plate Detection Demo")


def process_plate_image(plate_crop):
    if flip_image:
        plate_crop = cv2.flip(plate_crop, 1)

    h, w = plate_crop.shape[:2]
    # Gray Scale and Resize
    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    processed_plate = cv2.resize(gray_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return processed_plate

def run_ocr(processed_plate):
    try:
        result = ocr.run(processed_plate)
        text = ""
        conf = 0.0
        if isinstance(result, list) and len(result) > 0: 
            text = result[0]
        elif isinstance(result, str):
            text = result
            
        # If text found
        if text:
            conf = 1.0
            text = str(text).upper()
            text = re.sub(r'[^A-Z0-9 ]', '', text)
            
        return text, conf
    except Exception as e:
        return None, 0.0


if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Run YOLO (Car Detection)
        results = model(image, conf=conf_threshold)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                # Filter for vehicle
                if class_name in ['car']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw Car Box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{class_name} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    
                    # Crop Car for Plate Detection
                    car_img = image[y1:y2, x1:x2]
                    if car_img.size > 0:
                        plate_results = plate_model(car_img, conf=conf_threshold, verbose=False)
                        
                        if plate_results and len(plate_results[0].boxes) > 0:
                            pbox = plate_results[0].boxes[0]
                            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                            
                            # Calculate absolute coordinates for the plate
                            abs_px1 = x1 + px1
                            abs_py1 = y1 + py1
                            abs_px2 = x1 + px2
                            abs_py2 = y1 + py2
                            
                            # Draw Plate Box (Red)
                            cv2.rectangle(image, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 0, 255), 2)
                            
                            # Run OCR
                            plate_crop = car_img[py1:py2, px1:px2]
                            if plate_crop.size > 0:
                                processed = process_plate_image(plate_crop)
                                text, ocr_conf = run_ocr(processed)
                                
                                if text:
                                    label = f"{text}"
                                    cv2.putText(image, label, (abs_px1, abs_py1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                                    st.success(f"Detected: {text}")

        # Display Final Image
        st.image(image, channels="BGR", caption="Processed Result", width='stretch')

# --- VIDEO INPUT MODE ---
elif input_type == "Video":
    uploaded_file = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        stop_button = st.button("Stop Processing")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO
            results = model(frame, conf=conf_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    if class_name in ['car']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw Car
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                        
                        # Plate Detection
                        car_img = frame[y1:y2, x1:x2]
                        if car_img.size > 0:
                            plate_results = plate_model(car_img, conf=conf_threshold, verbose=False)
                            if plate_results and len(plate_results[0].boxes) > 0:
                                pbox = plate_results[0].boxes[0]
                                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                                
                                abs_px1, abs_py1 = x1 + px1, y1 + py1
                                abs_px2, abs_py2 = x1 + px2, y1 + py2
                                
                                cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 0, 255), 2)
                                
                                # OCR
                                plate_crop = car_img[py1:py2, px1:px2]
                                if plate_crop.size > 0:
                                    processed = process_plate_image(plate_crop)
                                    text, ocr_conf = run_ocr(processed)
                                    if text:
                                        cv2.putText(frame, f"{text}", (abs_px1, abs_py1 - 10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                                        st_frame.image(frame, channels="BGR", width='stretch')
        cap.release()