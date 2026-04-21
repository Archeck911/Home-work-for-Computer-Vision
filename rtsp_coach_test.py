# rtsp_coach_test.py
# ------------------
# RTSP -> YOLOv8 (person) -> простий трекер -> один COACH у кадрі.
# Вікно показу зафіксоване на 1920x1080.
# Логування старту/фінішу тренувань у sessions.txt

import os, sys, time, datetime
import cv2
import numpy as np
from math import hypot

# --- шляхи до твого проєкту
PROJECT_DIR = r"C:\coach_data\reid_full_train"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from recognizer_init import build_recognizer   # використовує embedder.pt + gallery_coach.npy

# --- твій RTSP
VIDEO_PATH = r"C:\coach_data\reid_full_train\TEST1.mp4"

# --- display size (фіксований розмір виводу)
DISPLAY_W, DISPLAY_H = 1920, 1080

# --- лог у файл
LOG_FILE = "sessions.txt"
session_active = False
session_start_time = None

# --- YOLOv8
try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Ultralytics YOLO не встановлено. Запусти: pip install ultralytics") from e

# --- простий трекер
class SimpleTracker:
    def __init__(self, max_lost=20):
        self.next_id = 1
        self.tracks = {}
        self.max_lost = max_lost

    @staticmethod
    def bbox_center(xyxy):
        x1,y1,x2,y2 = xyxy
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        w = max(0, inter_x2 - inter_x1); h = max(0, inter_y2 - inter_y1)
        inter = w * h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter + 1e-9
        return inter / union

    def update(self, detections):
        assigned = set()
        for tid, tr in list(self.tracks.items()):
            best_j, best_score = -1, -1e9
            tcx, tcy = tr["cx"], tr["cy"]
            for j, det in enumerate(detections):
                if j in assigned:
                    continue
                iou_val = self.iou(tr["xyxy"], det)
                cx, cy = self.bbox_center(det)
                dist = hypot(cy - tcy, cx - tcx)
                score = iou_val - (dist / 500.0)
                if score > best_score:
                    best_score, best_j = score, j
            if best_j >= 0 and best_score > -0.2:
                det = detections[best_j]
                cx, cy = self.bbox_center(det)
                tr["xyxy"] = det; tr["cx"] = cx; tr["cy"] = cy; tr["lost"] = 0
                assigned.add(best_j)
            else:
                tr["lost"] += 1
                if tr["lost"] > self.max_lost:
                    del self.tracks[tid]
        for j, det in enumerate(detections):
            if j in assigned:
                continue
            cx, cy = self.bbox_center(det)
            self.tracks[self.next_id] = {
                "xyxy": det, "cx": cx, "cy": cy, "lost": 0,
                "raw_score": 0.0, "is_pos": False
            }
            self.next_id += 1
        return self.tracks

# --- політика "один COACH"
COACH_MARGIN = 0.03
PROMOTE_NEED = 5
COACH_MISS   = 20

coach_id = None
coach_best = 0.0
promote_counter = 0
coach_miss_count = 0

def main():
    global coach_id, coach_best, promote_counter, coach_miss_count
    global session_active, session_start_time

    recognizer = build_recognizer()
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Не вдалося відкрити RTSP: {RTSP_URL}")

    tracker = SimpleTracker()
    frame_id = 0
    infer_every = 2

    WIN_NAME = "RTSP Coach Test (press q to quit)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, DISPLAY_W, DISPLAY_H)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("RTSP read fail, retry...")
            time.sleep(0.5)
            continue

        H, W = frame.shape[:2]

        results = model.predict(source=frame, imgsz=640, conf=0.35, verbose=False)
        dets_xyxy = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = int(b.cls.item())
                if cls != 0: 
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
                if (y2 - y1) >= 100 and (x2 - x1) >= 50:
                    dets_xyxy.append([x1, y1, x2, y2])

        tracks = tracker.update(dets_xyxy)

        per_frame = []
        if frame_id % infer_every == 0:
            for tid, tr in tracks.items():
                x1,y1,x2,y2 = map(int, tr["xyxy"])
                if (y2 - y1) < 100 or (x2 - x1) < 50:
                    tr["raw_score"] = 0.0
                    tr["is_pos"] = False
                    continue
                crop = frame[y1:y2, x1:x2]
                is_pos, score = recognizer.is_coach_crop(crop)
                tr["raw_score"] = float(score)
                tr["is_pos"] = bool(is_pos)
                per_frame.append((tid, float(score), bool(is_pos)))
        else:
            for tid, tr in tracks.items():
                per_frame.append((tid, float(tr.get("raw_score", 0.0)), bool(tr.get("is_pos", False))))

        if per_frame:
            pos_candidates = [(tid, sc) for tid, sc, pos in per_frame if pos]
            if not pos_candidates:
                coach_miss_count += 1
                if coach_miss_count > COACH_MISS:
                    coach_id = None
                    coach_best = 0.0
                promote_counter = 0
            else:
                tid_top, sc_top = max(pos_candidates, key=lambda x: x[1])
                if coach_id is None:
                    coach_id = tid_top
                    coach_best = sc_top
                    coach_miss_count = 0
                    promote_counter = 0
                else:
                    coach_miss_count = 0
                    if tid_top == coach_id:
                        coach_best = max(coach_best*0.9 + sc_top*0.1, sc_top)
                        promote_counter = 0
                    else:
                        if sc_top >= coach_best + COACH_MARGIN:
                            promote_counter += 1
                            if promote_counter >= PROMOTE_NEED:
                                coach_id = tid_top
                                coach_best = sc_top
                                promote_counter = 0
                        else:
                            promote_counter = 0

        # --- логіка початку/кінця тренування ---
        now = datetime.datetime.now()
        if coach_id is not None and not session_active:
            session_active = True
            session_start_time = now
            msg = f"[START] {session_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            print(msg)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        elif coach_id is None and session_active:
            session_active = False
            session_end_time = now
            duration = (session_end_time - session_start_time).total_seconds() / 60.0
            msg = (f"[END] {session_end_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                   f"Тривалість: {duration:.1f} хв")
            print(msg)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        # --- малювання
        for tid, tr in tracks.items():
            x1,y1,x2,y2 = map(int, tr["xyxy"])
            if tid == coach_id:
                label = "COACH"
                score = float(tr.get("raw_score", 0.0))
                color = (0, 255, 0)
            else:
                label = "OTHER"
                score = float(tr.get("raw_score", 0.0))
                color = (0, 128, 255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID {tid} {label} {score:.2f}", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN_NAME, out)

        frame_id += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
