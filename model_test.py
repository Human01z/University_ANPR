import os
import cv2
import easyocr
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# 1. Setup Models
# This downloads the specific YOLOv11 model you found
model_path = hf_hub_download(repo_id="morsetechlab/yolov11-license-plate-detection", 
                             filename="license-plate-finetune-v1s.pt")
model = YOLO(model_path)
reader = easyocr.Reader(['en'], gpu=False) # Set to True if you have an NVIDIA GPU

# 2. Path to your test image
image_path = "test.png" 

if os.path.exists(image_path):
    results = model.predict(source=image_path, conf=0.4)
    img = cv2.imread(image_path)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the plate
            plate_crop = img[y1:y2, x1:x2]
            
            # OCR with a "join" for two-line plates
            ocr_res = reader.readtext(plate_crop)
            # Combine all detected text lines into one string
            plate_text = " ".join([res[1] for res in ocr_res]).upper()
            
            print(f"✅ Detected Plate: {plate_text}")

            # Draw results on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the final result
    cv2.imwrite("final_detection.png", img)
    print("Saved result to final_detection.png")
else:
    print(f"Error: {image_path} not found in the current folder!")
    