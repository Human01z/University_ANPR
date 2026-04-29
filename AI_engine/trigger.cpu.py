import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

try:
    import serial
except Exception:
    serial = None

# ---------- Config ----------
RTSP_URL = "rtsp://USER:PASS@192.168.0.100:554/stream1"
ARDUINO_PORT = "COM3"  # Windows: COMx, Linux: /dev/ttyUSB0
ARDUINO_BAUD = 9600
ARDUINO_TRIGGER_WORDS = {"TRIGGER", "1", "VEHICLE"}
REQUIRE_VEHICLE_ON_TRIGGER = True  # ignore accidental beam triggers (e.g., pedestrians)

YOLO_MODEL = "v3.pt"
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
YOLO_CONF = 0.12
YOLO_IMGSZ = 960

RECORD_DIR = Path("captures_cpu")
RECORD_SECONDS = 6
RECORD_FPS = 15
RECORD_CODEC = "mp4v"

# CPU protection knobs
PROCESS_EVERY_N_FRAMES = 3
OCR_UPSCALE = 2.0

# Optional OSD/timestamp suppression
MASK_TOP_LEFT_OVERLAY = True
OVERLAY_MASK_WIDTH_RATIO = 0.35
OVERLAY_MASK_HEIGHT_RATIO = 0.14

SHOW_WINDOW = True
WINDOW_NAME = "ANPR Trigger CPU"


def suppress_camera_overlay(frame: np.ndarray) -> np.ndarray:
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


def init_serial():
    if serial is None:
        print("[warn] pyserial not installed; Arduino trigger disabled.")
        return None
    try:
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.05)
        time.sleep(2)
        print(f"[ok] Arduino connected on {ARDUINO_PORT}")
        return ser
    except Exception as e:
        print(f"[warn] Arduino not connected: {e}")
        return None


def read_trigger(ser) -> Tuple[bool, str]:
    if ser is None:
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
        imgsz=YOLO_IMGSZ,
        device="cpu",
        verbose=False,
    )
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if cls_id in VEHICLE_CLASS_IDS:
                return True
    return False


def detect_best_plate(model: YOLO, ocr: PaddleOCR, frame: np.ndarray) -> Tuple[str, float]:
    ai_frame = suppress_camera_overlay(frame)
    h, w = ai_frame.shape[:2]
    best_plate, best_conf = "NOT_FOUND", 0.0

    results = model.predict(
        source=ai_frame,
        conf=YOLO_CONF,
        iou=0.55,
        imgsz=YOLO_IMGSZ,
        device="cpu",
        verbose=False,
    )

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

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            up = cv2.resize(gray, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)
            ocr_res = ocr.ocr(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR), cls=True)
            if not ocr_res or not ocr_res[0]:
                continue

            text = "".join(seg[1][0] for seg in ocr_res[0])
            conf = float(np.mean([seg[1][1] for seg in ocr_res[0]]))
            plate = clean_sl_text(text)
            if plate != "NOT_FOUND" and conf > best_conf:
                best_plate, best_conf = plate, conf

    return best_plate, best_conf


def put_status(frame: np.ndarray, msg: str, color=(0, 255, 0)):
    cv2.rectangle(frame, (8, 8), (560, 50), (0, 0, 0), -1)
    cv2.putText(frame, msg, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def run_record_session(
    cap: cv2.VideoCapture,
    model: YOLO,
    ocr: PaddleOCR,
    clip_path: Path,
) -> Tuple[Optional[str], float, int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*RECORD_CODEC), RECORD_FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {clip_path}")

    best_plate, best_conf = "NOT_FOUND", 0.0
    frame_count = 0
    end_at = time.time() + RECORD_SECONDS

    while time.time() < end_at:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame_count += 1
        writer.write(frame)

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            plate, conf = detect_best_plate(model, ocr, frame)
            if plate != "NOT_FOUND" and conf > best_conf:
                best_plate, best_conf = plate, conf

        if SHOW_WINDOW:
            vis = frame.copy()
            put_status(vis, f"RECORDING... best={best_plate} conf={best_conf:.2f}", (0, 215, 255))
            cv2.imshow(WINDOW_NAME, vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    writer.release()
    return best_plate, best_conf, frame_count


def main():
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CPU models...")
    model = YOLO(YOLO_MODEL)
    ocr = PaddleOCR(lang="en", show_log=False, use_angle_cls=True, use_gpu=False)

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {RTSP_URL}")

    ser = init_serial()
    print("Armed. Waiting for Arduino trigger...")
    print("Press 'q' in preview window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            if SHOW_WINDOW:
                vis = frame.copy()
                put_status(vis, "ARMED: waiting for trigger", (0, 255, 0))
                cv2.imshow(WINDOW_NAME, vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            triggered, raw_line = read_trigger(ser)
            if not triggered:
                if raw_line:
                    print(f"[serial] Ignored line: {raw_line}")
                continue

            if REQUIRE_VEHICLE_ON_TRIGGER and not has_vehicle_in_frame(model, frame):
                print(f"[trigger] {raw_line} ignored (no vehicle seen in frame).")
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            clip_path = RECORD_DIR / f"trigger_{ts}.mp4"
            print(f"[trigger] Recording clip: {clip_path}")

            plate, conf, total = run_record_session(cap, model, ocr, clip_path)
            print(f"[done] frames={total}, best_plate={plate}, conf={conf:.3f}, saved={clip_path}")
    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
