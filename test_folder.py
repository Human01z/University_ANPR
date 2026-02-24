import os
import cv2
import easyocr
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# 1. Setup Models
model_path = hf_hub_download(repo_id="morsetechlab/yolov11-license-plate-detection", 
                             filename="license-plate-finetune-v1l.pt")
model = YOLO(model_path)
reader = easyocr.Reader(['en'], gpu=False)

input_folder = "samples"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

print(f"🚀 Starting upgraded inference on folder: {input_folder}...")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        # FIX 1: Lowered confidence to 0.25 to catch the missing 28%
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Padding (Kept small so we don't accidentally read bumper stickers)
                pad = 5
                plate_crop = img[max(0, y1-pad):min(img.shape[0], y2+pad), 
                                 max(0, x1-pad):min(img.shape[1], x2+pad)]
                
                # FIX 2: Advanced Pre-processing for OCR
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Bilateral filter keeps letter edges sharp while removing background noise
                blur = cv2.bilateralFilter(gray, 11, 17, 17)
                
                # Adaptive Thresholding handles uneven lighting and shadows
                binary_plate = cv2.adaptiveThreshold(blur, 255, 
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                     cv2.THRESH_BINARY, 11, 2)
                
                # FIX 3: Force EasyOCR to ONLY read uppercase letters and numbers
                valid_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ocr_res = reader.readtext(binary_plate, allowlist=valid_chars)
                
                # Combine detected text
                plate_text = " ".join([res[1] for res in ocr_res]).upper()
                
                # Visualizing
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, plate_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                print(f"📸 {filename} -> Detected: {plate_text}")

        save_path = os.path.join(output_folder, f"result_{filename}")
        cv2.imwrite(save_path, img)

print(f"\n✅ Done! Check the '{output_folder}' folder for your new results.")