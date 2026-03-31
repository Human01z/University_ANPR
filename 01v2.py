
import os
# --- MANDATORY STABILITY FIXES ---
os.environ['FLAGS_use_mkldnn'] = '0' 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
# ---------------------------------

import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

INPUT_DIR = "samples"
OUTPUT_DIR = "results_refined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Models Load
model = YOLO("v3.pt")
ocr = PaddleOCR(lang='en', show_log=False)

def fix_character_confusion(text):
    """Fixes common OCR mix-ups (e.g., reading an S as a 5)."""
    confused = {'8': 'B', '0': 'O', '1': 'I', '5': 'S', '2': 'Z', '6': 'G'}
    return "".join([confused.get(c, c) for c in text])

def clean_sl_text(raw_str):
    text = re.sub(r"[^A-Z0-9]", "", raw_str.upper())
    
    digits_found = re.findall(r'\d{4}', text)
    if not digits_found:
        return "NOT_FOUND"
    
    num_part = digits_found[-1]
    prefix_raw = text[:text.find(num_part)]
    
    # FIX 1: Fix character confusion BEFORE removing noise! 
    # (So if OCR reads '5G', we turn it to 'SG' before looking for the 'SG' noise)
    prefix_raw = fix_character_confusion(prefix_raw)
    
    # FIX 2: Added more province codes
    noise = ["SRILANKA", "SRI", "LANKA", "WP", "CP", "SP", "NW", "NC", "UP", "SG", "VA", "EP", "NP"]
    for n in noise:
        prefix_raw = prefix_raw.replace(n, "")
    
    letters = "".join(re.findall(r'[A-Z]', prefix_raw))
    
    # FIX 3: Dynamic fallback instead of forcing "LP"
    if len(letters) == 0:
        prefix = "??" # No letters found at all
    elif len(letters) > 3:
        prefix = letters[-3:] # Too many letters, grab the last 3
    else:
        prefix = letters # Perfect amount (1 to 3 letters)
        
    return f"{prefix}-{num_part}"

print("🎯 Running Precision Tuning with Floating Tags...")

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue
    
    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img is None: continue
    
    results = model.predict(source=img, conf=0.20, verbose=False)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = img.shape[:2]
            
            crop = img[max(0, y1-20):min(h, y2+20), max(0, x1-40):min(w, x2+40)]
            if crop.size == 0: continue
            
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            contrast = clahe.apply(gray)
            resized = cv2.resize(contrast, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(resized, -1, kernel)
            
            ocr_res = ocr.ocr(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR), cls=False)
            
            if ocr_res and ocr_res[0]:
                combined_text = "".join([res[1][0] for res in ocr_res[0]])
                candidate = clean_sl_text(combined_text)
                
                if candidate != "NOT_FOUND":
                    # --- NEW UI: Floating Tag ---
                    # 1. Draw the box around the plate
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # 2. Draw a solid green background block for the text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_scale = 0.8
                    text_thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(candidate, font, text_scale, text_thickness)
                    
                    # Make sure the text box doesn't go off the top of the screen
                    rect_y1 = max(y1 - text_h - 10, 0)
                    rect_y2 = max(y1, text_h + 10)
                    
                    cv2.rectangle(img, (x1, rect_y1), (x1 + text_w + 10, rect_y2), (0, 255, 0), -1)
                    
                    # 3. Write black text on top of the green block
                    cv2.putText(img, candidate, (x1 + 5, rect_y2 - 5), font, text_scale, (0, 0, 0), text_thickness)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"floating_{fname}"), img)
    print(f"✅ Processed: {fname}")

print("🏁 Tuning Complete. Check the 'results_refined' folder!")
