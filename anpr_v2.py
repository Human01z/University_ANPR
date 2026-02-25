import argparse
import os
import re
import cv2
import easyocr
from typing import List, Tuple
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# 1. PERMANENT PATH FIX - Finds the folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TEXT_CLEAN_RE = re.compile(r"[^A-Z0-9]")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
PLATE_PATTERN = re.compile(r"^[A-Z0-9]{5,11}$")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANPR System Core v2")
    # Using BASE_DIR so it works no matter where you run it from
    parser.add_argument("--input", default=os.path.join(BASE_DIR, "samples"))
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "results"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ocr-conf", type=float, default=0.15)
    parser.add_argument("--gpu", action="store_true", help="Use --gpu if you have NVIDIA hardware")
    parser.add_argument("--model", default="license-plate-finetune-v1l.pt")
    parser.add_argument("--repo-id", default="morsetechlab/yolov11-license-plate-detection")
    return parser.parse_args()

def clean_text(raw: str) -> str:
    cleaned = TEXT_CLEAN_RE.sub("", raw.upper())
    if len(cleaned) >= 5:
        prefix, suffix = cleaned[:-4], cleaned[-4:]
        char_to_num = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7', 'A': '4'}
        num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T', '4': 'A'}
        fixed_prefix = "".join([num_to_char.get(c, c) if c.isdigit() else c for c in prefix])
        fixed_suffix = "".join([char_to_num.get(c, c) if c.isalpha() else c for c in suffix])
        cleaned = fixed_prefix + fixed_suffix
    return cleaned

def pad_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = max(2, int(0.08 * bw)), max(2, int(0.20 * bh))
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)

def build_variants(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return [gray, clahe, otsu, adaptive]

def best_ocr_text(reader, crop, threshold):
    best_t, best_c = "", 0.0
    for var in build_variants(crop):
        res = reader.readtext(var, allowlist=VALID_CHARS)
        if not res: continue
        combined = "".join([t for _, t, _ in res])
        cleaned = clean_text(combined)
        avg_c = sum([c for _, _, c in res]) / len(res)
        if cleaned and avg_c >= threshold and PLATE_PATTERN.match(cleaned):
            if avg_c > best_c: best_c, best_t = float(avg_c), cleaned
    return best_t, best_c

def main():
    # --- 2. THE MANAGER TAKES THE ORDER ---
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Folder not found at {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)

    # --- 3. DOWNLOAD & LOAD MODELS ---
    print("⏳ Loading AI Models...")
    m_path = hf_hub_download(repo_id=args.repo_id, filename=args.model)
    model = YOLO(m_path)
    reader = easyocr.Reader(['en'], gpu=args.gpu)

    images = [f for f in os.listdir(args.input) if f.lower().endswith(IMAGE_EXTS)]
    print(f"🚀 Processing {len(images)} images in {args.input}...")

    # --- 4. THE EXECUTION LOOP ---
    for fname in images:
        img = cv2.imread(os.path.join(args.input, fname))
        h, w = img.shape[:2]
        results = model.predict(source=img, conf=args.conf, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, w, h)
                crop = img[y1:y2, x1:x2]
                
                txt, conf = best_ocr_text(reader, crop, args.ocr_conf)
                label = f"{txt} ({conf:.2f})" if txt else "UNREADABLE"
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"📸 {fname} -> {label}")

        cv2.imwrite(os.path.join(args.output, f"res_{fname}"), img)
    
    print(f"\n✅ All images processed! Check the folder: {args.output}")

# --- 5. THE TRIGGER ---
if __name__ == "__main__":
    main()