This project is a License Plate Recognition (LPR) system that uses a two-stage detection approach combined with OCR (Optical Character Recognition) to identify vehicle license plates.

Step 1: Car Detection. The system first uses a YOLO model (yolo11n.pt) to detect cars in an image or video frame. This narrows down the search area and improves accuracy by only looking for plates where vehicles are found.

Step 2: Plate Detection. Once a car is detected, the system crops that region and runs a second specialized YOLO model (plate_yolo11.pt) to locate the license plate within the car.

Step 3: OCR Processing. The detected plate is then preprocessed (converted to grayscale and upscaled for clarity) and passed to FastPlateOCR, which reads the text on the plate. The result is cleaned up to remove any invalid characters.

Step 4: Output. The recognized plate text is displayed on the image/video and can be saved to a database with a timestamp for logging purposes (e.g., entry/exit gate monitoring).

The project includes three main components:

main.py - Real-time video processing with car tracking and database logging
evaluate.py - Manual evaluation tool to test detection on a folder of images one by one
app.py - A Streamlit web interface for uploading and testing images/videos
trainv2.py - For pre-training the model