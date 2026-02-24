
import argparse, os, time, queue, threading
from datetime import datetime
import cv2, easyocr
from ultralytics import YOLO
import re




# Config
IMG_SIZE = 320
CONF_THRESHOLD = 0.35
OCR_CONF_THRESHOLD = 0.25
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
PLATE_ALLOWED = re.compile(r"[^A-Z0-9-]")
PROCESS_FPS = 4  # desired inference updates per second

def normalize_plate(text):
    return PLATE_ALLOWED.sub("", text.upper()).strip("-")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "yolov8n.pt"))
    return p.parse_args()

def camera_capture_thread(cap, frame_q, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        # always put latest frame, discarding old if queue full
        try:
            frame_q.put_nowait(frame)
        except queue.Full:
            # drop old frame and put newest
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                pass
    # release handled by main

def inference_thread(frame_q, overlay_q, stop_event, model_path):
    yolo = YOLO(model_path)
    ocr = easyocr.Reader(["en"], gpu=False)
    last_seen = {}
    last_process_time = 0
    min_interval = 1.0 / PROCESS_FPS
    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.2)
        except queue.Empty:
            continue
        now = time.time()
        if now - last_process_time < min_interval:
            # skip heavy inference to respect target FPS
            overlay_q.put(("frame", frame))
            continue
        last_process_time = now

        # run detection on downscaled frame for speed
        results = yolo(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        # annotate as in previous logic
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id not in VEHICLE_CLASS_IDS or conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
            x2 = max(1, min(x2, w)); y2 = max(1, min(y2, h))
            if x2<=x1 or y2<=y1: continue
            crop = frame[y1:y2, x1:x2]
            plate_text = ""
            # OCR only when high enough detection confidence
            if conf >= 0.6:
                ocr_result = ocr.readtext(crop)
                texts = [normalize_plate(r[1]) for r in ocr_result if r[2] >= OCR_CONF_THRESHOLD]
                texts = [t for t in texts if len(t) >= 4]
                plate_text = texts[0] if texts else ""
            cv2.rectangle(overlay, (x1,y1),(x2,y2),(0,255,0),2)
            if plate_text:
                now_dt = datetime.now()
                last_time = last_seen.get(plate_text)
                if not last_time or (now_dt - last_time).total_seconds() >= 3:
                    print(f"Plate: {plate_text} | Time: {now_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    last_seen[plate_text] = now_dt
                cv2.putText(overlay, plate_text, (x1, min(h-10,y2+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        overlay_q.put(("overlay", overlay))
    # cleanup if needed

def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError("Model not found: " + args.model)

    # open camera (try DirectShow first)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    frame_q = queue.Queue(maxsize=2)
    overlay_q = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    cap_thread = threading.Thread(target=camera_capture_thread, args=(cap, frame_q, stop_event), daemon=True)
    inf_thread = threading.Thread(target=inference_thread, args=(frame_q, overlay_q, stop_event, args.model), daemon=True)

    cap_thread.start()
    inf_thread.start()

    print("Starting webcam. Press 'q' to quit.")
    last_overlay = None
    try:
        while True:
            # show the most recent overlay if available, else show the last captured frame
            try:
                typ, data = overlay_q.get(timeout=0.1)
                if typ == "overlay":
                    last_overlay = data
                elif typ == "frame":
                    # we could show raw frame temporarily if no overlay
                    if last_overlay is None:
                        last_overlay = data
            except queue.Empty:
                pass

            if last_overlay is not None:
                cv2.imshow("ANPR Webcam Prototype", last_overlay)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
    finally:
        stop_event.set()
        time.sleep(0.2)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    main()