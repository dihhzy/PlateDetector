import cv2
import time
import os
from ultralytics import YOLO
import numpy as np
import re
from fast_plate_ocr import LicensePlateRecognizer

#------------------------------ DB

import sqlite3
from datetime import datetime

# Change the section id to EXIT_GATE if used at exit point
SECTION_ID = "ENTRY_GATE" 

DATA_DIR = '../data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DB_PATH = os.path.join(DATA_DIR, 'license_plates.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  track_id INTEGER,
                  plate_text TEXT,
                  timestamp TEXT,
                  image_path TEXT,
                  plate_image BLOB,
                  section_id TEXT)''') 
    conn.commit()
    conn.close()
    print("Database initialized")

init_db()


MODEL_DIR = '../models'
OUTPUT_DIR = '../output'

output_folder = os.path.join(OUTPUT_DIR, 'cars')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

plate_output_folder = os.path.join(OUTPUT_DIR, 'plates')
if not os.path.exists(plate_output_folder):
    os.makedirs(plate_output_folder)

ocr_output_folder = os.path.join(OUTPUT_DIR, 'result')
if not os.path.exists(ocr_output_folder):
    os.makedirs(ocr_output_folder)

model = YOLO(os.path.join(MODEL_DIR, 'yolo11n.pt'))
plate_model = YOLO(os.path.join(MODEL_DIR, 'plate_yolo11.pt'))


print("Initializing FastPlateOCR Pipeline")
try:
    #cuda for gpu, rasp use cpu
    ocr = LicensePlateRecognizer('cct-s-v1-global-model', device='cuda')
#   ocr = LicensePlateRecognizer('cct-s-v1-global-model', device='cpu')
    print("FastPlateOCR Model Loaded Successfully")
except Exception as e:
    print(f"Error loading FastPlateOCR: {e}")
    ocr = None

# for using esp
#esp32_url = 'http://192.168.0.59:80/stream' 
#print(f"Connecting to ESP32-CAM at {esp32_url}...")
#cap = cv2.VideoCapture(esp32_url, cv2.CAP_FFMPEG)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

# use video instead
cap = cv2.VideoCapture('example.mp4')

if not cap.isOpened():
    print("Error")
    exit()
else:
    print("Video stream opened successfully.")


track_history = {}
captured_ids = set()
CAR_CLASS_ID = 2 
DETECTION_THRESHOLD = 2.0


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, verbose=False)

    boxes = results[0].boxes
    
    current_time = time.time()
    current_frame_ids = set()

    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        cls_indices = boxes.cls.int().cpu().tolist()
        xyxys = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for box, track_id, cls, car_conf in zip(xyxys, track_ids, cls_indices, confidences):
            # Check if detected object is a car
            if cls == CAR_CLASS_ID:
                current_frame_ids.add(track_id)
                
                # Start timer and duration
                if track_id not in track_history:
                    track_history[track_id] = current_time
                
                duration = current_time - track_history[track_id]
                
                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id} {duration:.1f}s, {car_conf:.2f}",(x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                
                car_img = frame[y1:y2, x1:x2]

                best_plate_crop = None

                if car_img.size > 0:
                    plate_results = plate_model(car_img, conf=0.6, verbose=False)
                    if plate_results and len(plate_results[0].boxes) > 0:
                        pbox = plate_results[0].boxes[0]
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])

                        abs_x1 = x1 + px1
                        abs_y1 = y1 + py1
                        abs_x2 = x1 + px2
                        abs_y2 = y1 + py2

                        cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)

                        best_plate_crop = car_img[py1:py2, px1:px2]

                # If detected > 2 seconds and not yet captured
                if duration > DETECTION_THRESHOLD and track_id not in captured_ids:
                    # Crop the car image
                    car_img = frame[y1:y2, x1:x2]
                    if car_img.size > 0:
                        filename = os.path.join(output_folder, f"car_{track_id}_{int(current_time)}.jpg")
                        cv2.imwrite(filename, car_img)
                        print(f"Captured car ID {track_id} to {filename}")
                        
                        # Detect license plate within the cropped car image
                        plate_results = plate_model(car_img, conf=0.6, verbose=False)
                        
                        # Check if any plate is detected
                        if plate_results and len(plate_results[0].boxes) > 0:
                            pbox = plate_results[0].boxes[0]
                            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                            
                            # Crop plate from the car image
                            plate_crop = car_img[py1:py2, px1:px2]
                            
                            if plate_crop.size > 0:
                                plate_filename = os.path.join(plate_output_folder, f"plate_{track_id}_{int(current_time)}.jpg")
                                cv2.imwrite(plate_filename, plate_crop)
                                print(f"Captured plate for car ID {track_id}")

                                # --- OCR PROCESSING ---
                                print(f"Running OCR on plate ID {track_id}...")
                                try:
                                    if ocr is None:
                                        raise Exception("OCR Model not loaded")

                                    h, w = plate_crop.shape[:2]

                                    clean_plate = plate_crop[0:int(h * 0.75), :]

                                    #if camera is upside down
                                    #clean_plate = cv2.flip(clean_plate, 1)
                                    
                                    processed_plate = cv2.resize(clean_plate, (128, 64), interpolation=cv2.INTER_CUBIC)
                                    
                                    res_filename = os.path.join(ocr_output_folder, f"result_{track_id}.jpg")
                                    cv2.imwrite(res_filename, processed_plate)
                                    # ---------------------------------

                                    print("Performing OCR using FastPlateOCR...")
                                    
                                    result = ocr.run(processed_plate)
                                    
                                    text = ""
                                    
                                    if isinstance(result, list):
                                        if len(result) > 0:
                                            text = result[0]
                                    elif isinstance(result, str):
                                        text = result
                                    
                                    if text:
                                        text = str(text).upper()
                                        text = re.sub(r'[^A-Z0-9 ]', '', text)
                                    
                                    print(f"Result: {text}")
                                    
                                    if text:
                                        print(f"Plate: {text}")
                                        
                                        # Save to text file
                                        with open(os.path.join(ocr_output_folder, f"plate_{track_id}.txt"), "a") as f:
                                            f.write(f"{text}\n")

                                        # --- Save to Database ---
                                        try:
                                            conn = sqlite3.connect(DB_PATH)
                                            c = conn.cursor()
                                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            
                                            success, encoded_img = cv2.imencode('.jpg', plate_crop)
                                            if success:
                                                img_blob = encoded_img.tobytes()
                                            else:
                                                img_blob = None

                                            # Insert 0.0 for confidence placeholder
                                            c.execute("INSERT INTO detections (track_id, plate_text, timestamp, image_path, plate_image, section_id) VALUES (?, ?, ?, ?, ?, ?)",
                                                      (track_id, text, timestamp, plate_filename, img_blob, SECTION_ID))
                                            
                                            conn.commit()
                                            conn.close()
                                            print(f"Saved to Database (Section: {SECTION_ID})")
                                        except Exception as db_e:
                                            print(f"Database Error: {db_e}")
                                        # ----------------------------------

                                    else:
                                        print(f"No text found on plate ID {track_id}")

                                except Exception as e:
                                    print(f"OCR Failed for ID {track_id}: {e}")

                        captured_ids.add(track_id)

    ids_to_remove = [tid for tid in track_history if tid not in current_frame_ids]
    for tid in ids_to_remove:
        del track_history[tid]

    display_frame = cv2.resize(frame, (1280, 720)) 
    cv2.imshow("Entry Gate Monitor", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()