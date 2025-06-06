import cv2
import numpy as np
import torch
import os
from torchvision.ops import nms, batched_nms
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import SimpleNamespace
from ultralytics.utils.metrics import box_iou
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ROOT_DIR = os.getcwd()
WEIGHTS_PATH = os.path.join(ROOT_DIR, "yolov8x_tuned_c_new_dataset.pt")
VIDEO_PATH = os.path.join(ROOT_DIR, "test/test_4.mp4")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
OUTPUT_PATH_TRACKED = os.path.join(OUTPUT_DIR, "test_4_tracked.mp4")
OUTPUT_PATH_HEATMAP = os.path.join(OUTPUT_DIR, "test_4_heatmap.mp4")
OUTPUT_PATH_HIGHLIGHT = os.path.join(OUTPUT_DIR, "test_4_highlight.mp4")
FPS = 24
NEW_WIDTH, NEW_HEIGHT = 1920, 1080
MODEL_INPUT_SIZE = (1920, 1080)
SLICE_SIZE = (640, 640)
OVERLAP_RATIO = (0.2, 0.2)
CLASSES = ["car", "truck"]
HEATMAP_BLUR_KSIZE = (51, 51)
HEATMAP_RADIUS = 7
HEATMAP_ALPHA = 0.6
HEATMAP_BETA = 0.4
HIGHLIGHT_IDS = [44, 31, 2, 12, 15, 37]
DARK_ALPHA = 0.3

REID_INPUT_SIZE = (128, 128)
EMBEDDING_DIM   = 128

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_detection_model(
    weights_path: str,
    confidence_threshold: float = 0.5,
    device: str = "cpu",
    image_size: int = 640
):
    """Load YOLOv8 model and wrap it for SAHI slicing."""
    yolo = YOLO(weights_path)
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model=yolo,
        confidence_threshold=confidence_threshold,
        device=device,
        image_size=image_size,
    )

def init_tracker(frame_rate: int = FPS):
    """Initialize BYTETracker with default hyperparameters."""
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


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize to model input and convert to 3‑channel BGR grayscale."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, MODEL_INPUT_SIZE)
    return cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

def filter_detections_with_nms(result, iou_thresh: float = 0.5):
    boxes = torch.tensor(
        [det.bbox.to_xyxy() for det in result],
        dtype=torch.float32,
    )
    scores = torch.tensor(
        [det.score.value for det in result],
        dtype=torch.float32,
    )
    classes = torch.tensor(
        [det.category.id for det in result],
        dtype=torch.int64,
    )

    keep = batched_nms(boxes, scores, classes, iou_threshold=iou_thresh)
    truck_idx = (classes == 1).nonzero(as_tuple=True)[0]
    car_idx   = (classes == 0).nonzero(as_tuple=True)[0]

    keep_trucks = nms(boxes[truck_idx], scores[truck_idx], iou_thresh)
    keep_cars   = nms(boxes[car_idx],   scores[car_idx],   iou_thresh)

    kept_car_boxes   = boxes[car_idx[keep_cars]]
    kept_truck_boxes = boxes[truck_idx[keep_trucks]]
    ious = box_iou(kept_car_boxes, kept_truck_boxes)
    if len(ious[0]) > 0:
        car_keep_mask = ious.max(dim=1)[0] < iou_thresh
        keep_cars = keep_cars[car_keep_mask]

        keep = torch.cat([truck_idx[keep_trucks], car_idx[keep_cars]])

        keep = keep.sort().values.tolist()
        filtered = [result[i] for i in keep]
        return filtered
    
    return result


def detect_objects(
    small_frame: np.ndarray,
    detection_model,
) -> list:
    """Run SAHI sliced prediction and return filtered detections."""
    result = get_sliced_prediction(
        image=small_frame,
        detection_model=detection_model,
        slice_height=SLICE_SIZE[0],
        slice_width=SLICE_SIZE[1],
        overlap_height_ratio=OVERLAP_RATIO[0],
        overlap_width_ratio=OVERLAP_RATIO[1],
    )
    print(result.durations_in_seconds)
    return result.object_prediction_list


def prepare_track_input(detections: list):
    """Convert SAHI detections to tracker-friendly format."""
    xywh, conf, cls = [], [], []
    for det in detections:
        x1, y1, x2, y2 = det.bbox.to_xyxy()
        w, h = x2 - x1, y2 - y1
        xc, yc = x1 + w / 2, y1 + h / 2
        xywh.append([xc, yc, w, h])
        conf.append(det.score.value)
        cls.append(det.category.id)
    track_results = SimpleNamespace(
        xywh=torch.tensor(xywh, dtype=torch.float32),
        conf=torch.tensor(conf, dtype=torch.float32),
        cls=torch.tensor(cls, dtype=torch.float32),
    )
    return track_results


def update_heatmap(heatmap: np.ndarray, targets: list):
    """Draw center points of each tracked target onto the heatmap."""
    for t in targets:
        x1, y1, x2, y2 = map(int, t[:4])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= cx < NEW_WIDTH and 0 <= cy < NEW_HEIGHT:
            cv2.circle(heatmap, (cx, cy), HEATMAP_RADIUS, (255, 0, 0), -1)


def draw_tracks(frame: np.ndarray, targets: list):
    """Overlay bounding boxes, IDs, confidences, and class names."""
    for t in targets:
        x1, y1, x2, y2, tid, score, cls_id, _ = t
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID: {int(tid)} Cf: {score:.2f} Cl: {CLASSES[int(cls_id)]}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def apply_heatmap_overlay(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Create a colored, blurred heatmap and overlay it on the frame."""
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
    """
    Return a copy of `frame` that is darkened everywhere except inside
    bounding boxes whose track ID is in `highlight_ids`.
    """
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
    """Main processing loop: detection, tracking, heatmap, and output writing."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap_fps = max(cv2.CAP_PROP_FPS, FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_trk = cv2.VideoWriter(OUTPUT_PATH_TRACKED, fourcc, cap_fps, (NEW_WIDTH, NEW_HEIGHT))
    out_hm = cv2.VideoWriter(OUTPUT_PATH_HEATMAP, fourcc, cap_fps, (NEW_WIDTH, NEW_HEIGHT))
    out_hl = cv2.VideoWriter(OUTPUT_PATH_HIGHLIGHT, fourcc, cap_fps, (NEW_WIDTH, NEW_HEIGHT))
    det_model = load_detection_model(WEIGHTS_PATH, confidence_threshold=0.6)
    tracker = init_tracker(cap_fps)

    heatmap = np.zeros((NEW_HEIGHT, NEW_WIDTH), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
        small = preprocess_frame(frame)
        raw_dets = detect_objects(small, det_model)
        dets = filter_detections_with_nms(raw_dets)
        track_input = prepare_track_input(dets)
        online_targets = tracker.update(track_input, img=small)

        draw_tracks(frame, online_targets)
        update_heatmap(heatmap, online_targets)
        highlight_frame = spotlight_frame(frame, online_targets, HIGHLIGHT_IDS, DARK_ALPHA)
        
        out_trk.write(frame)
        out_hm.write(apply_heatmap_overlay(frame.copy(), heatmap))
        out_hl.write(highlight_frame)

    cap.release()
    out_trk.release()
    out_hm.release()
    out_hl.release()
    print("Processing complete.")


if __name__ == "__main__":
    process_video()
