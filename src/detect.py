import cv2
import numpy as np
import torch
import os
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import SimpleNamespace
from ultralytics.utils.metrics import box_iou
import time

ROOT_DIR = os.getcwd()
WEIGHTS_PATH = os.path.join(ROOT_DIR, "yolov8x_tuned_c_new_dataset.pt")
VIDEO_PATH = os.path.join(ROOT_DIR, "test/test_4.mp4")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output_detect")
OUTPUT_PATH_TRACKED = os.path.join(OUTPUT_DIR, "test_4_tracked.mp4")
OUTPUT_PATH_HEATMAP = os.path.join(OUTPUT_DIR, "test_4_heatmap.mp4")
OUTPUT_PATH_HIGHLIGHT = os.path.join(OUTPUT_DIR, "test_4_highlight.mp4")
FPS = 24
NEW_WIDTH, NEW_HEIGHT = 1920, 1080

SLICE_SIZE = 640
OVERLAP_RATIO = 0.2
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3

CLASSES = ["car", "truck"]
HEATMAP_BLUR_KSIZE = (51, 51)
HEATMAP_RADIUS = 7
HEATMAP_ALPHA = 0.6
HEATMAP_BETA = 0.4
HIGHLIGHT_IDS = [44]
DARK_ALPHA = 0.3

MOTION_THRESHOLD = 1.2
STATIC_DETECTION_INTERVAL = 6
GAUSSIAN_KERNEL_SIZE = 21
MOTION_HISTORY_SIZE = 3

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class MotionDetector:
    def __init__(self, slice_size: int, motion_threshold: float = MOTION_THRESHOLD):
        self.slice_size = slice_size
        self.motion_threshold = motion_threshold
        self.prev_frames = {}
        self.motion_history = {}

    def detect_motion_in_slices(self, frame: np.ndarray, slices: list) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_map = {}

        for i, (x1, y1, x2, y2) in enumerate(slices):
            slice_key = f"{x1}_{y1}_{x2}_{y2}"
            current_slice = gray[y1:y2, x1:x2]

            if slice_key in self.prev_frames:
                prev_slice = self.prev_frames[slice_key]

                diff = cv2.absdiff(current_slice, prev_slice)
                blurred = cv2.GaussianBlur(diff, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
                motion_score = np.mean(blurred)

                has_motion = motion_score > self.motion_threshold

                if slice_key not in self.motion_history:
                    self.motion_history[slice_key] = []

                self.motion_history[slice_key].append(has_motion)
                if len(self.motion_history[slice_key]) > MOTION_HISTORY_SIZE:
                    self.motion_history[slice_key].pop(0)

                recent_motion = any(self.motion_history[slice_key])
                motion_map[slice_key] = {
                    'has_motion': recent_motion,
                    'motion_score': motion_score,
                    'slice_coords': (x1, y1, x2, y2)
                }
            else:
                motion_map[slice_key] = {
                    'has_motion': True,
                    'motion_score': 0.0,
                    'slice_coords': (x1, y1, x2, y2)
                }

            self.prev_frames[slice_key] = current_slice.copy()

        return motion_map


class CustomSAHI:
    def __init__(self, model_path: str, slice_size: int = 640, overlap_ratio: float = 0.2,
                 confidence_threshold: float = 0.6):
        print(f"[INFO] Initializing YOLO model from: {model_path}")
        start_time = time.time()
        self.yolo_model = YOLO(model_path)
        load_time = time.time() - start_time
        print(f"[INFO] Model loaded in {load_time:.2f}s")

        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold
        self.step = int(slice_size * (1 - overlap_ratio))

        self.motion_detector = MotionDetector(slice_size)
        self.cached_detections = {}
        self.frame_count = 0

        print(f"[INFO] SAHI parameters: slice_size={slice_size}, overlap_ratio={overlap_ratio}, step={self.step}")
        print(
            f"[INFO] Motion detection enabled: threshold={MOTION_THRESHOLD}, static_interval={STATIC_DETECTION_INTERVAL}")

    def generate_slices(self, image_height: int, image_width: int):
        slices = []

        for y in range(0, image_height, self.step):
            for x in range(0, image_width, self.step):
                x_end = min(x + self.slice_size, image_width)
                y_end = min(y + self.slice_size, image_height)

                if (x_end - x) < self.slice_size // 2 or (y_end - y) < self.slice_size // 2:
                    continue

                slices.append((x, y, x_end, y_end))

        return slices

    def detect_on_slice(self, image: np.ndarray, slice_coords: tuple):
        x1, y1, x2, y2 = slice_coords
        slice_img = image[y1:y2, x1:x2]

        results = self.yolo_model(slice_img, conf=self.confidence_threshold, verbose=False)

        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())

                global_x1 = box[0] + x1
                global_y1 = box[1] + y1
                global_x2 = box[2] + x1
                global_y2 = box[3] + y1

                detections.append({
                    'bbox': [global_x1, global_y1, global_x2, global_y2],
                    'confidence': float(conf),
                    'class_id': cls
                })

        return detections

    def predict(self, image: np.ndarray):
        self.frame_count += 1
        h, w = image.shape[:2]
        slices = self.generate_slices(h, w)

        motion_map = self.motion_detector.detect_motion_in_slices(image, slices)

        all_detections = []
        slices_processed = 0
        slices_with_motion = 0
        slices_cached = 0

        for slice_coords in slices:
            x1, y1, x2, y2 = slice_coords
            slice_key = f"{x1}_{y1}_{x2}_{y2}"

            if slice_key in motion_map:
                motion_info = motion_map[slice_key]
                has_motion = motion_info['has_motion']

                should_detect = (has_motion or
                                 self.frame_count % STATIC_DETECTION_INTERVAL == 0 or
                                 slice_key not in self.cached_detections)

                if should_detect:
                    slice_detections = self.detect_on_slice(image, slice_coords)
                    self.cached_detections[slice_key] = slice_detections
                    slices_processed += 1
                    if has_motion:
                        slices_with_motion += 1
                else:
                    slice_detections = self.cached_detections.get(slice_key, [])
                    slices_cached += 1

                all_detections.extend(slice_detections)

        if all_detections:
            merged_detections = self.merge_detections_with_nms(all_detections)

            if self.frame_count % 50 == 0:
                print(f"[MOTION] Frame {self.frame_count}: Processed {slices_processed}/{len(slices)} slices")
                print(f"         Motion slices: {slices_with_motion}, Cached: {slices_cached}")

            return merged_detections, motion_map

        return []

    def merge_detections_with_nms(self, detections: list):
        if not detections:
            return []

        cars = [d for d in detections if d['class_id'] == 0]
        trucks = [d for d in detections if d['class_id'] == 1]

        merged_detections = []

        for class_detections in [cars, trucks]:
            if not class_detections:
                continue

            boxes = torch.tensor([d['bbox'] for d in class_detections], dtype=torch.float32)
            scores = torch.tensor([d['confidence'] for d in class_detections], dtype=torch.float32)

            keep_indices = nms(boxes, scores, NMS_THRESHOLD)

            for idx in keep_indices:
                merged_detections.append(class_detections[idx])

        if len([d for d in merged_detections if d['class_id'] == 0]) > 0 and \
                len([d for d in merged_detections if d['class_id'] == 1]) > 0:

            car_detections = [d for d in merged_detections if d['class_id'] == 0]
            truck_detections = [d for d in merged_detections if d['class_id'] == 1]

            car_boxes = torch.tensor([d['bbox'] for d in car_detections], dtype=torch.float32)
            truck_boxes = torch.tensor([d['bbox'] for d in truck_detections], dtype=torch.float32)

            ious = box_iou(car_boxes, truck_boxes)

            cars_to_keep = []
            for i, car_det in enumerate(car_detections):
                max_iou = ious[i].max().item() if len(ious[i]) > 0 else 0
                if max_iou < NMS_THRESHOLD:
                    cars_to_keep.append(car_det)

            merged_detections = truck_detections + cars_to_keep

        return merged_detections


def init_tracker(frame_rate: int = FPS):
    print(f"[INFO] Initializing BYTETracker with frame_rate={frame_rate}")
    args = SimpleNamespace(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.5,
        track_buffer=60,
        match_thresh=0.9,
        aspect_ratio_thresh=3,
        min_box_area=15,
        gmc=True,
        use_byte=True,
        fuse_score=True,
    )
    return BYTETracker(args, frame_rate=frame_rate)


def prepare_track_input(detections: list):
    xywh, conf, cls = [], [], []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w, h = x2 - x1, y2 - y1
        xc, yc = x1 + w / 2, y1 + h / 2
        xywh.append([xc, yc, w, h])
        conf.append(det['confidence'])
        cls.append(det['class_id'])

    track_results = SimpleNamespace(
        xywh=torch.tensor(xywh, dtype=torch.float32),
        conf=torch.tensor(conf, dtype=torch.float32),
        cls=torch.tensor(cls, dtype=torch.float32),
    )
    return track_results


def update_heatmap(heatmap: np.ndarray, targets: list):
    for t in targets:
        x1, y1, x2, y2 = map(int, t[:4])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= cx < NEW_WIDTH and 0 <= cy < NEW_HEIGHT:
            cv2.circle(heatmap, (cx, cy), HEATMAP_RADIUS, (255, 0, 0), -1)


def draw_tracks(frame: np.ndarray, targets: list, motion_map: dict = None):
    if motion_map is not None:
        overlay = frame.copy()
        for info in motion_map.values():
            x1, y1, x2, y2 = info['slice_coords']
            if not info['has_motion']:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        alpha = 0.4
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    for t in targets:
        x1, y1, x2, y2, tid, score, cls_id, _ = t
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID: {int(tid)} Cf: {score:.2f} Cl: {CLASSES[int(cls_id)]}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



def apply_heatmap_overlay(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(heatmap, HEATMAP_BLUR_KSIZE, 0)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, HEATMAP_ALPHA, colored, HEATMAP_BETA, 0)


def spotlight_frame(
        frame: np.ndarray,
        targets: list,
        highlight_ids: list[int],
        dark_alpha: float = 0.3
) -> np.ndarray:
    dark = (frame.astype(np.float32) * dark_alpha).astype(np.uint8)

    H, W = frame.shape[:2]
    for t in targets:
        tid = int(t[4])
        if tid in highlight_ids:
            x1, y1, x2, y2 = map(int, t[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            dark[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    return dark


def process_video():
    print("=" * 60)
    print("[INFO] Starting video processing with adaptive detection...")
    print("=" * 60)

    processing_start = time.time()

    print(f"[INFO] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("[ERROR] Could not open video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap_fps = max(original_fps, FPS)

    print(f"[INFO] Video info: {total_frames} frames, {original_fps:.2f} FPS")
    print(f"[INFO] Output resolution: {NEW_WIDTH}x{NEW_HEIGHT}")
    print(f"[INFO] Processing FPS: {cap_fps}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print(f"[INFO] Creating output files...")
    out_trk = cv2.VideoWriter(OUTPUT_PATH_TRACKED, fourcc, cap_fps, (NEW_WIDTH, NEW_HEIGHT))
    out_hm = cv2.VideoWriter(OUTPUT_PATH_HEATMAP, fourcc, cap_fps, (NEW_WIDTH, NEW_HEIGHT))
    out_hl = cv2.VideoWriter(OUTPUT_PATH_HIGHLIGHT, fourcc, cap_fps, (NEW_WIDTH, NEW_HEIGHT))

    sahi_detector = CustomSAHI(
        model_path=WEIGHTS_PATH,
        slice_size=SLICE_SIZE,
        overlap_ratio=OVERLAP_RATIO,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    tracker = init_tracker(cap_fps)
    heatmap = np.zeros((NEW_HEIGHT, NEW_WIDTH), dtype=np.float32)

    print("\n" + "=" * 60)
    print("[INFO] Starting frame processing...")
    print("=" * 60)

    frame_count = 0
    total_detection_time = 0
    total_tracking_time = 0
    total_visualization_time = 0

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        print(f"\n[FRAME {frame_count:04d}/{total_frames}]", end=" ")

        resize_start = time.time()
        frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
        resize_time = time.time() - resize_start

        detection_start = time.time()
        detections, motion_map = sahi_detector.predict(frame)
        detection_time = time.time() - detection_start
        total_detection_time += detection_time

        tracking_start = time.time()
        track_input = prepare_track_input(detections)
        online_targets = tracker.update(track_input, img=frame)
        tracking_time = time.time() - tracking_start
        total_tracking_time += tracking_time

        visualization_start = time.time()
        draw_tracks(frame, online_targets, motion_map=motion_map)
        update_heatmap(heatmap, online_targets)
        highlight_frame = spotlight_frame(frame, online_targets, HIGHLIGHT_IDS, DARK_ALPHA)

        out_trk.write(frame)
        out_hm.write(apply_heatmap_overlay(frame.copy(), heatmap))
        out_hl.write(highlight_frame)
        visualization_time = time.time() - visualization_start
        total_visualization_time += visualization_time

        frame_total_time = time.time() - frame_start

        print(f"Detections: {len(detections):2d} | Tracks: {len(online_targets):2d}")
        print(
            f"         Times: Resize: {resize_time * 1000:5.1f}ms | Detection: {detection_time * 1000:6.1f}ms | Tracking: {tracking_time * 1000:5.1f}ms | Viz: {visualization_time * 1000:5.1f}ms | Total: {frame_total_time * 1000:6.1f}ms")

        fps_current = 1.0 / frame_total_time if frame_total_time > 0 else 0
        progress = (frame_count / total_frames) * 100
        print(f"         FPS: {fps_current:5.1f} | Progress: {progress:5.1f}%")

        if frame_count % 100 == 0:
            avg_detection = (total_detection_time / frame_count) * 1000
            avg_tracking = (total_tracking_time / frame_count) * 1000
            avg_visualization = (total_visualization_time / frame_count) * 1000
            avg_fps = frame_count / (time.time() - processing_start)

            cached_slices = len(sahi_detector.cached_detections)

            print(f"\n[STATS] Processed {frame_count} frames")
            print(
                f"        Average times: Detection: {avg_detection:.1f}ms | Tracking: {avg_tracking:.1f}ms | Viz: {avg_visualization:.1f}ms")
            print(f"        Average FPS: {avg_fps:.2f} | Cached slices: {cached_slices}")

    processing_total_time = time.time() - processing_start

    cap.release()
    out_trk.release()
    out_hm.release()
    out_hl.release()

    print("\n" + "=" * 60)
    print("[INFO] Processing completed!")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {processing_total_time:.2f}s ({processing_total_time / 60:.1f} min)")
    print(f"Average FPS: {frame_count / processing_total_time:.2f}")
    print(f"Average detection time: {(total_detection_time / frame_count) * 1000:.1f}ms per frame")
    print(f"Average tracking time: {(total_tracking_time / frame_count) * 1000:.1f}ms per frame")
    print(f"Average visualization time: {(total_visualization_time / frame_count) * 1000:.1f}ms per frame")
    print(f"Total cached slices: {len(sahi_detector.cached_detections)}")
    print(f"\nOutput files:")
    print(f"  - Tracked: {OUTPUT_PATH_TRACKED}")
    print(f"  - Heatmap: {OUTPUT_PATH_HEATMAP}")
    print(f"  - Highlight: {OUTPUT_PATH_HIGHLIGHT}")
    print("=" * 60)


if __name__ == "__main__":
    process_video()