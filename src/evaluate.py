import cv2
import os
from ultralytics import YOLO
import numpy as np
import re
from fast_plate_ocr import LicensePlateRecognizer

# ---------------- CONFIGURATION ----------------
IMGS_DIR = 'imgs' 
MODEL_DIR = '../models'
CAR_CLASS_ID = 2 
# -----------------------------------------------

def evaluate():
    print(f"Loading models from {MODEL_DIR}...")
    try:
        model = YOLO(os.path.join(MODEL_DIR, 'yolo11n.pt'))
        plate_model = YOLO(os.path.join(MODEL_DIR, 'plate_yolo11.pt'))
    except Exception as e:
        print(f"Error loading YOLO models: {e}")
        return

    try:
        ocr = LicensePlateRecognizer('global-plates-mobile-vit-v2-model', device='cuda')
        print("FastPlateOCR Model Loaded Successfully")
    except Exception as e:
        print(f"Error loading FastPlateOCR: {e}")
        ocr = None

    if not os.path.exists(IMGS_DIR):
        print(f"Error: Directory '{IMGS_DIR}' not found.")
        return

    image_files = [f for f in os.listdir(IMGS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No images found in '{IMGS_DIR}'.")
        return

    print(f"Found {len(image_files)} images. Starting evaluation...")
    print("Controls: Press any key to next image, 'q' to quit.")

    for img_file in image_files:
        img_path = os.path.join(IMGS_DIR, img_file)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Could not read {img_file}")
            continue

        frame = cv2.flip(frame, 1)

        print(f"\n--- Processing: {img_file} ---")

        results = model(frame, verbose=False)
        
        car_detected = False
        processed_plate_centers = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                
                if cls == CAR_CLASS_ID:
                    car_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw Car Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Crop Car
                    car_img = frame[y1:y2, x1:x2]
                    
                    if car_img.size > 0:
                        # Detect Plate inside Car
                        plate_results = plate_model(car_img, conf=0.6, verbose=False)
                        
                        if plate_results and len(plate_results[0].boxes) > 0:
                            # Take the first/best plate found
                            pbox = plate_results[0].boxes[0]
                            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                            
                            # Calculate absolute coordinates for drawing on main frame
                            abs_x1 = x1 + px1
                            abs_y1 = y1 + py1
                            abs_x2 = x1 + px2
                            abs_y2 = y1 + py2
                            
                            # Calculate center of the detected plate
                            center_x = (abs_x1 + abs_x2) / 2
                            center_y = (abs_y1 + abs_y2) / 2
                            
                            is_duplicate = False
                            for (pcx, pcy) in processed_plate_centers:
                                dist = ((center_x - pcx)**2 + (center_y - pcy)**2)**0.5
                                if dist < 50: # If within 50 pixels of another plate, consider it a duplicate
                                    is_duplicate = True
                                    break
                            
                            if is_duplicate:
                                continue
                                
                            processed_plate_centers.append((center_x, center_y))

                            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                            
                            plate_crop = car_img[py1:py2, px1:px2]
                            
                            if plate_crop.size > 0 and ocr is not None:
                                h, w = plate_crop.shape[:2]
                                clean_plate = plate_crop[0:int(h * 0.95), :]
                                gray_plate = cv2.cvtColor(clean_plate, cv2.COLOR_BGR2GRAY)
                                processed_plate = cv2.resize(gray_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                                
                                ocr_result = ocr.run(processed_plate)
                                
                                text = ""
                                if isinstance(ocr_result, list) and len(ocr_result) > 0:
                                    text = ocr_result[0]
                                elif isinstance(ocr_result, str):
                                    text = ocr_result
                                
                                if text:
                                    text = str(text).upper()
                                    text = re.sub(r'[^A-Z0-9 ]', '', text)
                                    print(f"Plate Detected: {text}")
                                    
                                    cv2.putText(frame, text, (abs_x1, abs_y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                                else:
                                    print("Plate detected but OCR failed.")

        if not car_detected:
            print("No car detected. Attempting full-frame plate detection...")
            plate_results = plate_model(frame, conf=0.5, verbose=False)
            if plate_results and len(plate_results[0].boxes) > 0:
                for pbox in plate_results[0].boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    
                    plate_crop = frame[py1:py2, px1:px2]
                    if plate_crop.size > 0 and ocr is not None:
                        h, w = plate_crop.shape[:2]
                        clean_plate = plate_crop[0:int(h * 0.75), :]
                        gray_plate = cv2.cvtColor(clean_plate, cv2.COLOR_BGR2GRAY)
                        processed_plate = cv2.resize(gray_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        
                        ocr_result = ocr.run(processed_plate)
                        text = ""
                        if isinstance(ocr_result, list) and len(ocr_result) > 0:
                            text = ocr_result[0]
                        elif isinstance(ocr_result, str):
                            text = ocr_result
                        
                        if text:
                            text = str(text).upper()
                            text = re.sub(r'[^A-Z0-9 ]', '', text)
                            print(f"Plate Detected (Full Frame): {text}")
                            cv2.putText(frame, text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        display_h, display_w = frame.shape[:2]
        if display_w > 1280:
            scale = 1280 / display_w
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        cv2.imshow("Manual Evaluation", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Exiting evaluation.")
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluate()