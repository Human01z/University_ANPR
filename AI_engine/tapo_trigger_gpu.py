import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import List, Tuple
import urllib.request

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR

try:
    import serial
except Exception:
    serial = None

# ---------- Config ----------
RTSP_URL = "rtsp://USER:PASS@192.168.0.100:554/stream1"
ARDUINO_PORT = "COM3"          # e.g. COM3 on Windows, /dev/ttyUSB0 on Linux
ARDUINO_BAUD = 9600
ARDUINO_TRIGGER_WORDS = {"TRIGGER", "1", "VEHICLE"}
PYTHON_TRIGGER_COOLDOWN_SEC = 3.8  # guards against duplicate sensor bounce on host side
REQUIRE_VEHICLE_ON_TRIGGER = True  # ignore accidental beam triggers (e.g., pedestrians)

YOLO_MODEL = "v3.pt"
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

BURST_FRAMES = 12                # capture 10-15 frames per trigger
CAPTURE_GAP_SEC = 0.08           # ~12.5 fps capture

PORTAL_API_URL = "http://127.0.0.1:8000/api/events"
PORTAL_API_KEY = ""
DEFAULT_DIRECTION = "Entry"
DEFAULT_GATE = "Gate-1"

SAVE_DEBUG_FRAMES = True
DEBUG_DIR = "captures"

# Ignore timestamp/logo overlay often present at top-left of CCTV frames.
MASK_TOP_LEFT_OVERLAY = True
OVERLAY_MASK_WIDTH_RATIO = 0.35
OVERLAY_MASK_HEIGHT_RATIO = 0.14

# ---------- Helpers ----------
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sharpness_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def suppress_camera_overlay(frame: np.ndarray) -> np.ndarray:
    """Masks top-left camera OSD text to avoid false plate OCR reads."""
    if not MASK_TOP_LEFT_OVERLAY:
        return frame
    masked = frame.copy()
    h, w = masked.shape[:2]
    mw = max(1, int(w * OVERLAY_MASK_WIDTH_RATIO))
    mh = max(1, int(h * OVERLAY_MASK_HEIGHT_RATIO))
    cv2.rectangle(masked, (0, 0), (mw, mh), (0, 0, 0), thickness=-1)
    return masked


def clean_sl_text(raw: str) -> str:
    txt = re.sub(r"[^A-Z0-9]", "", raw.upper())
    m = re.findall(r"\d{4}", txt)
    if not m:
        return "NOT_FOUND"
    num = m[-1]
    idx = txt.rfind(num)
    letters = re.findall(r"[A-Z]", txt[:idx])
    if not letters:
        return f"??-{num}"
    return f"{''.join(letters)[-3:]}-{num}"


def encode_jpg_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def send_event(plate: str, conf: float, images: List[np.ndarray], vehicle_type: str = "vehicle"):
    images_b64 = [encode_jpg_b64(im) for im in images]
    images_b64 = [x for x in images_b64 if x]
    payload = {
        "plate_ai": plate,
        "ai_conf": float(conf),
        "direction": DEFAULT_DIRECTION,
        "gate": DEFAULT_GATE,
        "vehicle_type": vehicle_type,
        "event_time": utcnow_iso(),
        "images_base64": images_b64,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if PORTAL_API_KEY:
        headers["x-api-key"] = PORTAL_API_KEY

    req = urllib.request.Request(PORTAL_API_URL, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=8) as r:
        _ = r.read()


def init_serial():
    if serial is None:
        print("[warn] pyserial not installed; Arduino trigger disabled.")
        return None
    try:
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
        time.sleep(2)
        print(f"Arduino connected on {ARDUINO_PORT}")
        return ser
    except Exception as e:
        print(f"[warn] Arduino not connected: {e}")
        return None


def wait_for_trigger(ser) -> Tuple[bool, str]:
    # Arduino should send lines like: TRIGGER (matches the provided Uno sketch).
    if ser is None:
        # Fallback: manual keyboard trigger with OpenCV window not required; timed polling
        time.sleep(0.2)
        return False, ""
    line = ser.readline().decode(errors="ignore").strip().upper()
    if not line:
        return False, ""
    if line in ARDUINO_TRIGGER_WORDS:
        return True, line
    # Accept counter format: "TRIGGER 1", "TRIGGER 2", ...
    if re.match(r"^TRIGGER\s+\d+$", line):
        return True, line
    return False, line


def has_vehicle_in_frame(model: YOLO, frame: np.ndarray) -> bool:
    ai_frame = suppress_camera_overlay(frame)
    results = model.predict(
        source=ai_frame,
        conf=0.15,
        iou=0.55,
        imgsz=960,
        device=0 if torch.cuda.is_available() else "cpu",
        half=torch.cuda.is_available(),
        verbose=False,
    )
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if cls_id in VEHICLE_CLASS_IDS:
                return True
    return False


def capture_burst(cap: cv2.VideoCapture, n: int) -> List[np.ndarray]:
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
        time.sleep(CAPTURE_GAP_SEC)
    return frames


def detect_plate_in_frame(model: YOLO, ocr: PaddleOCR, frame: np.ndarray) -> Tuple[str, float, str]:
    ai_frame = suppress_camera_overlay(frame)
    h, w = ai_frame.shape[:2]
    best_plate, best_score, vtype = "NOT_FOUND", 0.0, "vehicle"

    results = model.predict(source=ai_frame, conf=0.12, iou=0.55, imgsz=1280, device=0 if torch.cuda.is_available() else "cpu", half=torch.cuda.is_available(), verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = ai_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            vtype = "motorcycle" if cls_id == 3 else "vehicle"

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            ocr_res = ocr.ocr(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR), cls=True)
            if not ocr_res or not ocr_res[0]:
                continue

            text = "".join(seg[1][0] for seg in ocr_res[0])
            conf = float(np.mean([seg[1][1] for seg in ocr_res[0]]))
            plate = clean_sl_text(text)
            if plate != "NOT_FOUND" and conf > best_score:
                best_plate, best_score = plate, conf

    return best_plate, best_score, vtype


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

    print("Loading YOLO + OCR...")
    model = YOLO(YOLO_MODEL)
    ocr = PaddleOCR(lang="en", show_log=False, use_angle_cls=True, use_gpu=torch.cuda.is_available())

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {RTSP_URL}")

    ser = init_serial()
    print("Waiting for Arduino trigger...")
    last_trigger_time = 0.0

    while True:
        ok, preview_frame = cap.read()
        if not ok or preview_frame is None:
            continue

        trig, raw_line = wait_for_trigger(ser)
        if not trig:
            if raw_line:
                print(f"[serial] Ignored line: {raw_line}")
            continue
        now = time.time()
        if (now - last_trigger_time) < PYTHON_TRIGGER_COOLDOWN_SEC:
            print("[serial] Trigger ignored due to Python cooldown.")
            continue
        if REQUIRE_VEHICLE_ON_TRIGGER and not has_vehicle_in_frame(model, preview_frame):
            print(f"[trigger] {raw_line} ignored (no vehicle seen in frame).")
            continue
        last_trigger_time = now

        print("Trigger received -> capturing burst")
        burst = capture_burst(cap, BURST_FRAMES)
        if not burst:
            continue

        frame_candidates = []
        for idx, frame in enumerate(burst):
            plate, conf, vtype = detect_plate_in_frame(model, ocr, frame)
            sharp = sharpness_score(frame)
            frame_candidates.append((plate, conf, sharp, vtype, frame))

        # Prefer best OCR confidence; tie-breaker by sharpness
        frame_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_plate, best_conf, _sharp, vtype, best_frame = frame_candidates[0]

        # Keep top 3 frames to help guard verification
        upload_frames = [x[4] for x in frame_candidates[:3]]

        if SAVE_DEBUG_FRAMES:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, fr in enumerate(upload_frames):
                cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_{i}_{best_plate}.jpg"), fr)

        try:
            send_event(best_plate, best_conf, upload_frames, vtype)
            print(f"Uploaded event -> plate={best_plate}, conf={best_conf:.3f}")
        except Exception as e:
            print(f"[upload error] {e}")


if __name__ == "__main__":
    main()
