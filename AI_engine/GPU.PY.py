import os
import re
import json
import csv
import time
import base64
import urllib.request
from collections import Counter, deque
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- Stability ---
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# --- Video configuration ---
VIDEO_INPUT = "sample.mp4"
VIDEO_OUTPUT = "anpr_output_gpu.mp4"
CSV_OUTPUT = "anpr_output_gpu.csv"
YOLO_MODEL = "v3.pt"
FRAME_SKIP = 1
YOLO_CONF = 0.08
YOLO_IOU = 0.55
YOLO_IMGSZ = 1536  # larger inference size for small motorbike plates
MIN_OCR_SCORE = 0.50
MIN_STABLE_VOTES = 3
EXTENDED_OCR_ON_LOW_CONF = True
LOW_CONF_RETRY_THRESHOLD = 0.65
USE_HALF_PRECISION = True
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# --- Web portal bridge ---
ENABLE_PORTAL_UPLOAD = True
PORTAL_API_URL = "http://127.0.0.1:8000/api/events"
PORTAL_API_KEY = ""
DEFAULT_GATE = "Gate-1"


def fix_character_confusion(text: str) -> str:
    confusion = {"8": "B", "0": "O", "1": "I", "5": "S", "2": "Z", "6": "G"}
    return "".join(confusion.get(c, c) for c in text)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_sl_text(raw_str: str) -> str:
    text = re.sub(r"[^A-Z0-9]", "", raw_str.upper())
    matches = re.findall(r"\d{4}", text)
    if not matches:
        return "NOT_FOUND"

    num_part = matches[-1]
    idx = text.rfind(num_part)
    prefix_raw = fix_character_confusion(text[:idx])

    noise = ["SRILANKA", "SRI", "LANKA", "WP", "CP", "SP", "NW", "NC", "UP", "SG", "VA", "EP", "NP"]
    for n in noise:
        prefix_raw = prefix_raw.replace(n, "")

    letters = "".join(re.findall(r"[A-Z]", prefix_raw))
    if not letters:
        return f"??-{num_part}"
    return f"{letters[-3:]}-{num_part}"


def get_rotation_code(cap: cv2.VideoCapture):
    orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    if orientation == 90:
        return cv2.ROTATE_90_CLOCKWISE
    if orientation == 180:
        return cv2.ROTATE_180
    if orientation == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def maybe_rotate(frame: np.ndarray, rotate_code):
    return frame if rotate_code is None else cv2.rotate(frame, rotate_code)


def _sort_and_limit(rois, limit=8):
    rois = [r for r in rois if r.size > 0 and r.shape[0] > 8 and r.shape[1] > 16]
    rois = sorted(rois, key=lambda im: im.shape[0] * im.shape[1], reverse=True)
    return rois[:limit]


def plate_candidates(vehicle_crop: np.ndarray, cls_id: int = -1):
    if vehicle_crop.size == 0:
        return []

    h, w = vehicle_crop.shape[:2]
    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 40, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw == 0 or ch == 0:
            continue
        aspect = cw / float(ch)
        area_ratio = (cw * ch) / float(w * h + 1e-6)
        if 1.8 <= aspect <= 7.5 and 0.008 <= area_ratio <= 0.35:
            px, py = max(3, cw // 10), max(3, ch // 5)
            rois.append(vehicle_crop[max(0, y - py):min(h, y + ch + py), max(0, x - px):min(w, x + cw + px)])

    rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)
    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    grad_x = (255 * (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min() + 1e-6)).astype("uint8")
    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    _, bw = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rect_k)

    contours2, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours2:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(ch + 1e-6)
        area_ratio = (cw * ch) / float(w * h + 1e-6)
        if 2.0 <= aspect <= 8.0 and 0.006 <= area_ratio <= 0.40:
            px, py = max(3, cw // 12), max(3, ch // 5)
            rois.append(vehicle_crop[max(0, y - py):min(h, y + ch + py), max(0, x - px):min(w, x + cw + px)])

    rois.append(vehicle_crop[int(h * 0.45):int(h * 0.95), int(w * 0.05):int(w * 0.98)])
    if cls_id == 3:
        rois.append(vehicle_crop[int(h * 0.20):int(h * 0.62), int(w * 0.08):int(w * 0.95)])
        rois.append(vehicle_crop[int(h * 0.52):int(h * 0.96), int(w * 0.08):int(w * 0.95)])
    rois.append(vehicle_crop)

    return _sort_and_limit(rois, limit=12)


def rotate_image(image: np.ndarray, angle_deg: float):
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def ocr_variants(img: np.ndarray, extended: bool = False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    up = cv2.resize(clahe, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(up, -1, kernel)

    th_adapt = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 6)
    _, th_otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_otsu_sharp = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variants = [
        cv2.cvtColor(up, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(th_adapt, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(th_otsu, cv2.COLOR_GRAY2BGR),
    ]

    if extended:
        variants.append(cv2.cvtColor(th_otsu_sharp, cv2.COLOR_GRAY2BGR))
        base = variants[0]
        variants.extend([rotate_image(base, -8), rotate_image(base, 8), rotate_image(base, -15), rotate_image(base, 15)])
    else:
        base = variants[0]
        variants.extend([rotate_image(base, -8), rotate_image(base, 8)])

    return variants


def read_plate_text(ocr: PaddleOCR, roi: np.ndarray):
    def run_pass(variants):
        best_text, best_conf = "NOT_FOUND", 0.0
        for variant in variants:
            ocr_res = ocr.ocr(variant, cls=True)
            if not ocr_res or not ocr_res[0]:
                continue

            raw = "".join(seg[1][0] for seg in ocr_res[0])
            mean_conf = float(np.mean([seg[1][1] for seg in ocr_res[0]]))
            candidate = clean_sl_text(raw)
            score = mean_conf + (0.1 if re.match(r"^[A-Z]{1,3}-\d{4}$", candidate) else 0.0)
            if candidate != "NOT_FOUND" and score > best_conf:
                best_text, best_conf = candidate, score
        return best_text, best_conf

    best_text, best_conf = run_pass(ocr_variants(roi, extended=False))
    if EXTENDED_OCR_ON_LOW_CONF and best_conf < LOW_CONF_RETRY_THRESHOLD:
        retry_text, retry_conf = run_pass(ocr_variants(roi, extended=True))
        if retry_conf > best_conf:
            best_text, best_conf = retry_text, retry_conf
    return best_text, best_conf


def write_csv_results(track_memory: dict, csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "plate_text", "best_score", "last_bbox"])
        for tid, data in track_memory.items():
            best_text = data.get("best_text", "NOT_FOUND")
            best_conf = data.get("best_conf", 0.0)
            if best_text == "NOT_FOUND":
                continue
            writer.writerow([tid, best_text, f"{best_conf:.4f}", data.get("box")])


def finalize_track_result(track_id: int, data: dict, final_results: dict):
    best_text = data.get("confirmed_text") or data.get("best_text", "NOT_FOUND")
    best_conf = max(data.get("confirmed_conf", 0.0), data.get("best_conf", 0.0))
    if best_text == "NOT_FOUND":
        return
    final_results[track_id] = {
        "best_text": best_text,
        "best_conf": best_conf,
        "box": data.get("box"),
    }


def upload_event_to_portal(plate_text: str, ai_conf: float, direction: str, vehicle_type: str, event_time: str, crop_bgr: np.ndarray):
    if not ENABLE_PORTAL_UPLOAD:
        return
    ok, buf = cv2.imencode(".jpg", crop_bgr)
    if not ok:
        return
    payload = {
        "plate_ai": plate_text,
        "ai_conf": float(ai_conf),
        "direction": direction,
        "gate": DEFAULT_GATE,
        "vehicle_type": vehicle_type,
        "event_time": event_time,
        "images_base64": [base64.b64encode(buf.tobytes()).decode("utf-8")],
    }
    headers = {"Content-Type": "application/json"}
    if PORTAL_API_KEY:
        headers["x-api-key"] = PORTAL_API_KEY
    req = urllib.request.Request(PORTAL_API_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=3)
    except Exception as e:
        print(f"[portal-upload] warning: {e}")


def is_plate_model(model: YOLO) -> bool:
    names = model.names if hasattr(model, "names") else {}
    for _idx, name in names.items() if isinstance(names, dict) else enumerate(names):
        label = str(name).lower()
        if any(k in label for k in ["plate", "licence", "license", "number plate", "registration"]):
            return True
    return False


def main():
    print("Loading models...")
    device = 0 if torch.cuda.is_available() else "cpu"
    use_half = bool(device == 0 and USE_HALF_PRECISION)

    model = YOLO(YOLO_MODEL)
    ocr = PaddleOCR(lang="en", show_log=False, use_angle_cls=True, use_gpu=bool(device == 0))

    start_time = time.time()
    model_is_plate = is_plate_model(model)

    print(f"Inference device: {'cuda:0' if device == 0 else 'cpu'} (half={use_half})")
    if model_is_plate:
        print("Detected plate-specific model.")
    else:
        print("Detected generic model (vehicle->plate ROI pipeline).")

    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {VIDEO_INPUT}")

    rotate_code = get_rotation_code(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = (in_h, in_w) if rotate_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE) else (in_w, in_h)

    out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    track_memory = {}
    final_results = {}
    next_id = 0
    frame_count = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = maybe_rotate(frame, rotate_code)
        h, w = frame.shape[:2]
        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            detections = []
            results = model.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ, device=device, half=use_half, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    if not model_is_plate and cls_id not in VEHICLE_CLASS_IDS:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    best_plate, best_conf = "NOT_FOUND", 0.0
                    if model_is_plate:
                        best_plate, best_conf = read_plate_text(ocr, crop)
                    else:
                        for roi in plate_candidates(crop, cls_id=cls_id):
                            plate, conf = read_plate_text(ocr, roi)
                            if conf > best_conf:
                                best_plate, best_conf = plate, conf

                    detections.append((x1, y1, x2, y2, best_plate, best_conf, cls_id))

            for x1, y1, x2, y2, plate, conf, cls_id in detections:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                matched, min_d = None, 10**9
                for tid, data in track_memory.items():
                    px, py = data["center"]
                    d = abs(px - cx) + abs(py - cy)
                    if d < min_d and d < 140:
                        min_d = d
                        matched = tid

                if matched is None:
                    matched = next_id
                    next_id += 1
                    track_memory[matched] = {"center": (cx, cy), "texts": deque(maxlen=12), "box": (x1, y1, x2, y2), "missed": 0, "best_text": "NOT_FOUND", "best_conf": 0.0, "confirmed_text": None, "confirmed_conf": 0.0, "sent_to_portal": False, "vehicle_type": "unknown"}

                track_memory[matched]["center"] = (cx, cy)
                track_memory[matched]["box"] = (x1, y1, x2, y2)
                track_memory[matched]["missed"] = 0
                track_memory[matched]["vehicle_type"] = "motorcycle" if cls_id == 3 else "vehicle"
                if plate != "NOT_FOUND" and conf >= MIN_OCR_SCORE:
                    track_memory[matched]["texts"].append((plate, conf))
                    if conf > track_memory[matched]["best_conf"]:
                        track_memory[matched]["best_conf"] = conf
                        track_memory[matched]["best_text"] = plate
                    weighted = {}
                    counts = Counter()
                    for txt, tconf in track_memory[matched]["texts"]:
                        weighted[txt] = weighted.get(txt, 0.0) + tconf
                        counts[txt] += 1
                    voted_text = max(weighted.items(), key=lambda kv: kv[1])[0]
                    if counts[voted_text] >= MIN_STABLE_VOTES:
                        track_memory[matched]["confirmed_text"] = voted_text
                        track_memory[matched]["confirmed_conf"] = weighted[voted_text] / counts[voted_text]

                if track_memory[matched].get("confirmed_text") and not track_memory[matched].get("sent_to_portal"):
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        upload_event_to_portal(
                            plate_text=track_memory[matched]["confirmed_text"],
                            ai_conf=track_memory[matched].get("confirmed_conf", 0.0),
                            direction="Entry",
                            vehicle_type=track_memory[matched].get("vehicle_type", "vehicle"),
                            event_time=utcnow_iso(),
                            crop_bgr=crop,
                        )
                        track_memory[matched]["sent_to_portal"] = True

            active_centers = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2, _, _ in detections]
            for tid in list(track_memory.keys()):
                tx, ty = track_memory[tid]["center"]
                near_any = any(abs(tx - cx) + abs(ty - cy) < 180 for cx, cy in active_centers)
                if not near_any:
                    track_memory[tid]["missed"] += 1
                if track_memory[tid]["missed"] > 20:
                    finalize_track_result(tid, track_memory[tid], final_results)
                    del track_memory[tid]

        for data in track_memory.values():
            x1, y1, x2, y2 = data["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            plate_text = "PLATE_NOT_CLEAR"
            if data.get("confirmed_text"):
                plate_text = data["confirmed_text"]
            elif data["texts"]:
                weighted, counts = {}, Counter()
                for txt, conf in data["texts"]:
                    weighted[txt] = weighted.get(txt, 0.0) + conf
                    counts[txt] += 1
                plate_text = max(weighted.items(), key=lambda kv: kv[1])[0]
                if counts[plate_text] < MIN_STABLE_VOTES:
                    plate_text = "PLATE_NOT_CLEAR"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(plate_text, font, 0.7, 2)
            by1, by2 = max(0, y1 - th - 10), y1
            cv2.rectangle(frame, (x1, by1), (x1 + tw + 10, by2), (0, 255, 0), -1)
            cv2.putText(frame, plate_text, (x1 + 5, by2 - 4), font, 0.7, (0, 0, 0), 2)

        out.write(frame)
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()

    for tid, data in track_memory.items():
        finalize_track_result(tid, data, final_results)

    write_csv_results(final_results, CSV_OUTPUT)
    elapsed = time.time() - start_time
    print(f"Done. Saved output video to {VIDEO_OUTPUT}")
    print(f"Saved best-per-vehicle CSV to {CSV_OUTPUT}")
    print(f"Total processing time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
