"""Low-duty-cycle YOLO object detection and presence tracking."""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


YOLO_MODEL_FILENAME = "yolov10n.onnx"
YOLO_MODEL_URL = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.onnx"
YOLO_MODEL_SHA256 = "7025ea1913f9a259cf8a8465ed608e10610d1bb376db2e0348b13e3bd286e0d3"
YOLO_INPUT_SIZE = 640
# Use confidence hysteresis: reliable detections open a recording, while a
# confirmed living object may be retained through a temporary confidence dip.
YOLO_TRACK_CONFIDENCE_THRESHOLD = 0.10
YOLO_INITIAL_CONFIDENCE_THRESHOLD = 0.30
YOLO_MOTION_BOX_CENTER_DELTA = 0.015
YOLO_MOTION_BOX_SIZE_CHANGE = 0.12
YOLO_PRESENCE_SAMPLE_COUNT = 5
YOLO_PRESENCE_SAMPLE_INTERVAL_SECONDS = 1.0
YOLO_PRESENCE_ZOOM_SAMPLE_COUNT = 2
YOLO_PRESENCE_MAX_TRACKED_CROPS = 4
YOLO_PRESENCE_MAX_ACTIVE_TRACKS = 16
YOLO_PRESENCE_CROP_MIN_WIDTH = 0.40
YOLO_PRESENCE_CROP_MIN_HEIGHT = 0.55
YOLO_PRESENCE_CROP_WIDTH_SCALE = 4.0
YOLO_PRESENCE_CROP_HEIGHT_SCALE = 2.5
YOLO_PRESENCE_FALLBACK_CROPS = (
    (0.00, 0.00, 0.55, 1.00),
    (0.45, 0.00, 0.55, 1.00),
)

COCO_LABELS = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

# People, transport and animals from the standard COCO label set.
ANIMAL_CLASS_IDS = frozenset(range(14, 24))
PERSISTENT_PRESENCE_CLASS_IDS = frozenset({0, *ANIMAL_CLASS_IDS})
DEFAULT_RELEVANT_CLASS_IDS = frozenset({*PERSISTENT_PRESENCE_CLASS_IDS, *range(1, 9)})

cv2 = None
np = None


@dataclass(frozen=True)
class ObjectDetection:
    class_id: int
    label: str
    confidence: float
    # Normalized x, y, width and height.
    box: tuple[float, float, float, float]


def require_opencv():
    global cv2
    if cv2 is not None:
        return cv2
    try:
        import cv2 as opencv_module
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for YOLO detection: "
            "python -m pip install opencv-python-headless"
        ) from error
    cv2 = opencv_module
    return cv2


def require_numpy():
    global np
    if np is not None:
        return np
    try:
        import numpy as numpy_module
    except ImportError as error:
        raise RuntimeError("NumPy is required for YOLO detection: python -m pip install numpy") from error
    np = numpy_module
    return np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_yolo_model(
    path: Path,
    *,
    url: str = YOLO_MODEL_URL,
    expected_sha256: str = YOLO_MODEL_SHA256,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    if path.exists():
        actual_hash = file_sha256(path)
        if actual_hash.casefold() != expected_sha256.casefold():
            raise RuntimeError(
                f"YOLO model checksum mismatch for {path}: "
                f"expected {expected_sha256}, got {actual_hash}"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    if log is not None:
        log(f"yolo_model_downloading url={url} path={path}")
    request = urllib.request.Request(url, headers={"User-Agent": "Watchdog/1.0"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60.0) as response, temporary_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash.casefold() != expected_sha256.casefold():
            raise RuntimeError(
                f"downloaded YOLO model checksum mismatch: "
                f"expected {expected_sha256}, got {actual_hash}"
            )
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    if log is not None:
        log(f"yolo_model_ready path={path}")


class YoloObjectDetector:
    """Thread-safe OpenCV DNN wrapper for the official YOLOv10n ONNX model."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = YOLO_TRACK_CONFIDENCE_THRESHOLD,
        relevant_class_ids: Iterable[int] = DEFAULT_RELEVANT_CLASS_IDS,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.relevant_class_ids = frozenset(int(value) for value in relevant_class_ids)
        self.log = log
        self.lock = threading.Lock()
        self.model_ready = False
        self.net = None

    def _ensure_model_locked(self) -> None:
        if self.model_ready:
            return
        ensure_yolo_model(self.model_path, log=self.log)
        self.model_ready = True

    def prepare(self) -> None:
        """Download and verify the model without loading the DNN into memory."""

        with self.lock:
            self._ensure_model_locked()

    def _load_locked(self) -> None:
        if self.net is not None:
            return
        opencv = require_opencv()
        self._ensure_model_locked()
        net = opencv.dnn.readNetFromONNX(str(self.model_path))
        self.net = net
        if self.log is not None:
            self.log(f"yolo_model_loaded path={self.model_path}")

    def detect(self, image) -> tuple[ObjectDetection, ...]:
        opencv = require_opencv()
        numpy = require_numpy()
        if image is None:
            raise ValueError("YOLO input image is missing")
        source = numpy.asarray(image, dtype="uint8")
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("YOLO input must be a BGR image")
        source_height, source_width = source.shape[:2]
        if source_width <= 0 or source_height <= 0:
            return ()

        scale = min(YOLO_INPUT_SIZE / source_width, YOLO_INPUT_SIZE / source_height)
        resized_width = max(1, min(YOLO_INPUT_SIZE, int(round(source_width * scale))))
        resized_height = max(1, min(YOLO_INPUT_SIZE, int(round(source_height * scale))))
        resized = opencv.resize(source, (resized_width, resized_height), interpolation=opencv.INTER_LINEAR)
        pad_x = (YOLO_INPUT_SIZE - resized_width) // 2
        pad_y = (YOLO_INPUT_SIZE - resized_height) // 2
        canvas = numpy.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype="uint8")
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        blob = opencv.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
            swapRB=True,
            crop=False,
        )

        with self.lock:
            self._load_locked()
            self.net.setInput(blob)
            output = self.net.forward()

        rows = numpy.asarray(output).reshape(-1, 6)
        detections = []
        for x1, y1, x2, y2, confidence, class_value in rows:
            confidence = float(confidence)
            class_id = int(round(float(class_value)))
            if confidence < self.confidence_threshold or class_id not in self.relevant_class_ids:
                continue
            source_x1 = max(0.0, min(float(source_width), (float(x1) - pad_x) / scale))
            source_y1 = max(0.0, min(float(source_height), (float(y1) - pad_y) / scale))
            source_x2 = max(0.0, min(float(source_width), (float(x2) - pad_x) / scale))
            source_y2 = max(0.0, min(float(source_height), (float(y2) - pad_y) / scale))
            if source_x2 <= source_x1 or source_y2 <= source_y1:
                continue
            label = COCO_LABELS[class_id] if 0 <= class_id < len(COCO_LABELS) else str(class_id)
            detections.append(
                ObjectDetection(
                    class_id,
                    label,
                    confidence,
                    (
                        source_x1 / source_width,
                        source_y1 / source_height,
                        (source_x2 - source_x1) / source_width,
                        (source_y2 - source_y1) / source_height,
                    ),
                )
            )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return tuple(detections)


def detect_presence_samples(
    detector: YoloObjectDetector,
    images: Iterable[object],
    tracked: Iterable[ObjectDetection] = (),
) -> tuple[ObjectDetection, ...]:
    """Use the newest recent frame in which a person or animal is visible.

    Presence checks are intentionally temporal: a single full-frame YOLO pass
    can miss seated or mutually occluding people even when adjacent frames are
    detected correctly.  The latest frame is always checked first.  Older
    samples are tried when the latest result has neither a reliable new living
    object nor a lower-confidence detection matching an existing track.  If
    every full-frame pass misses, recent frames are searched again in enlarged
    crops around known occupants.  The latest full-frame result remains
    authoritative for vehicle-only checks.
    """

    samples = tuple(images)
    tracked = tuple(tracked)
    if not samples:
        return ()

    def confirms_presence(detections: Iterable[ObjectDetection]) -> bool:
        return any(
            item.class_id in PERSISTENT_PRESENCE_CLASS_IDS
            and (
                item.confidence >= YOLO_INITIAL_CONFIDENCE_THRESHOLD
                or any(detections_match(previous, item) for previous in tracked)
            )
            for item in detections
        )

    latest_detections: tuple[ObjectDetection, ...] = ()
    for offset, image in enumerate(reversed(samples)):
        detections = detector.detect(image)
        if offset == 0:
            latest_detections = detections
        if confirms_presence(detections):
            return detections

    crop_regions = presence_search_regions(tracked)
    if crop_regions:
        zoom_samples = samples[-YOLO_PRESENCE_ZOOM_SAMPLE_COUNT:]
        for image in reversed(zoom_samples):
            for region in crop_regions:
                detections = detect_in_normalized_crop(detector, image, region)
                if confirms_presence(detections):
                    return detections
    return latest_detections


def presence_search_regions(
    tracked: Iterable[ObjectDetection],
) -> tuple[tuple[float, float, float, float], ...]:
    """Build zoomed search regions around the last known living-object boxes."""

    regions = []
    for item in tracked:
        if item.class_id not in PERSISTENT_PRESENCE_CLASS_IDS:
            continue
        x, y, width, height = item.box
        crop_width = min(
            1.0,
            max(YOLO_PRESENCE_CROP_MIN_WIDTH, width * YOLO_PRESENCE_CROP_WIDTH_SCALE),
        )
        crop_height = min(
            1.0,
            max(YOLO_PRESENCE_CROP_MIN_HEIGHT, height * YOLO_PRESENCE_CROP_HEIGHT_SCALE),
        )
        center_x = x + width * 0.5
        center_y = y + height * 0.5
        crop_x = max(0.0, min(1.0 - crop_width, center_x - crop_width * 0.5))
        crop_y = max(0.0, min(1.0 - crop_height, center_y - crop_height * 0.5))
        region = (crop_x, crop_y, crop_width, crop_height)
        if any(box_iou(previous, region) >= 0.80 for previous in regions):
            continue
        regions.append(region)
        if len(regions) >= YOLO_PRESENCE_MAX_TRACKED_CROPS:
            break

    if regions:
        regions.extend(YOLO_PRESENCE_FALLBACK_CROPS)
    return tuple(regions)


def detect_in_normalized_crop(
    detector: YoloObjectDetector,
    image,
    region: tuple[float, float, float, float],
) -> tuple[ObjectDetection, ...]:
    """Run detection on a crop and map its boxes back to the source frame."""

    numpy = require_numpy()
    source = numpy.asarray(image, dtype="uint8")
    if source.ndim != 3 or source.shape[2] != 3:
        return ()
    source_height, source_width = source.shape[:2]
    if source_width <= 0 or source_height <= 0:
        return ()
    region_x, region_y, region_width, region_height = region
    pixel_x0 = max(0, min(source_width - 1, int(round(region_x * source_width))))
    pixel_y0 = max(0, min(source_height - 1, int(round(region_y * source_height))))
    pixel_x1 = max(
        pixel_x0 + 1,
        min(source_width, int(round((region_x + region_width) * source_width))),
    )
    pixel_y1 = max(
        pixel_y0 + 1,
        min(source_height, int(round((region_y + region_height) * source_height))),
    )
    crop = source[pixel_y0:pixel_y1, pixel_x0:pixel_x1]
    actual_x = pixel_x0 / source_width
    actual_y = pixel_y0 / source_height
    actual_width = (pixel_x1 - pixel_x0) / source_width
    actual_height = (pixel_y1 - pixel_y0) / source_height
    return tuple(
        ObjectDetection(
            item.class_id,
            item.label,
            item.confidence,
            (
                actual_x + item.box[0] * actual_width,
                actual_y + item.box[1] * actual_height,
                item.box[2] * actual_width,
                item.box[3] * actual_height,
            ),
        )
        for item in detector.detect(crop)
    )


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def box_intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[0] + first[2], second[0] + second[2])
    y1 = min(first[1] + first[3], second[1] + second[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = box_intersection(first, second)
    union = box_area(first) + box_area(second) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def motion_overlaps_detection(
    detection: ObjectDetection,
    motion_regions: Iterable[tuple[float, float, float, float]],
) -> bool:
    for region in motion_regions:
        intersection = box_intersection(detection.box, region)
        smaller_area = min(box_area(detection.box), box_area(region))
        if smaller_area > 0.0 and intersection / smaller_area >= 0.20:
            return True
        center_x = region[0] + region[2] * 0.5
        center_y = region[1] + region[3] * 0.5
        if (
            detection.box[0] <= center_x <= detection.box[0] + detection.box[2]
            and detection.box[1] <= center_y <= detection.box[1] + detection.box[3]
        ):
            return True
    return False


def detections_match(first: ObjectDetection, second: ObjectDetection) -> bool:
    classes_match = first.class_id == second.class_id or (
        first.class_id in ANIMAL_CLASS_IDS and second.class_id in ANIMAL_CLASS_IDS
    )
    if not classes_match:
        return False
    if box_iou(first.box, second.box) >= 0.10:
        return True
    first_center = (first.box[0] + first.box[2] * 0.5, first.box[1] + first.box[3] * 0.5)
    second_center = (second.box[0] + second.box[2] * 0.5, second.box[1] + second.box[3] * 0.5)
    distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
    first_area = box_area(first.box)
    second_area = box_area(second.box)
    size_ratio = 0.0 if max(first_area, second_area) <= 0.0 else min(first_area, second_area) / max(first_area, second_area)
    return distance <= 0.25 and size_ratio >= 0.20


def detection_box_moved(first: ObjectDetection, second: ObjectDetection) -> bool:
    """Reject detector jitter while retaining meaningful transport movement."""

    first_center = (first.box[0] + first.box[2] * 0.5, first.box[1] + first.box[3] * 0.5)
    second_center = (second.box[0] + second.box[2] * 0.5, second.box[1] + second.box[3] * 0.5)
    center_delta = (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5
    first_area = box_area(first.box)
    second_area = box_area(second.box)
    size_change = 0.0
    if max(first_area, second_area) > 0.0:
        size_change = abs(first_area - second_area) / max(first_area, second_area)
    return (
        center_delta >= YOLO_MOTION_BOX_CENTER_DELTA
        or size_change >= YOLO_MOTION_BOX_SIZE_CHANGE
    )


class YoloPresenceTracker:
    """Require motion to open a segment, then track every living occupant."""

    def __init__(
        self,
        active: Iterable[ObjectDetection],
        background: Iterable[ObjectDetection],
        *,
        misses_to_stop: int = 2,
        persistent_misses_to_stop: int = 3,
        initial_confidence_threshold: float = YOLO_INITIAL_CONFIDENCE_THRESHOLD,
        persistent_confidence_threshold: float = YOLO_TRACK_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.active = list(active)
        self.background = list(background)
        self.misses_to_stop = max(1, int(misses_to_stop))
        self.persistent_misses_to_stop = max(
            self.misses_to_stop,
            int(persistent_misses_to_stop),
        )
        self.initial_confidence_threshold = float(initial_confidence_threshold)
        self.persistent_confidence_threshold = float(persistent_confidence_threshold)
        self.consecutive_misses = 0

    @classmethod
    def from_initial(
        cls,
        detections: Iterable[ObjectDetection],
        motion_regions: Iterable[tuple[float, float, float, float]],
        *,
        misses_to_stop: int = 2,
        persistent_misses_to_stop: int = 3,
        initial_confidence_threshold: float = YOLO_INITIAL_CONFIDENCE_THRESHOLD,
        persistent_confidence_threshold: float = YOLO_TRACK_CONFIDENCE_THRESHOLD,
    ) -> Optional["YoloPresenceTracker"]:
        detections = [
            item
            for item in detections
            if item.confidence >= initial_confidence_threshold
        ]
        regions = tuple(motion_regions)
        motion_associated = [item for item in detections if motion_overlaps_detection(item, regions)]
        if not motion_associated:
            return None

        # Motion association is deliberately required only to open a segment.
        # Once a person or animal has caused a recording to start, other living
        # occupants already visible in the room must also keep it open.  In
        # particular, a seated person can remain almost completely outside the
        # foreground region created by somebody else walking out of the room.
        active = list(motion_associated)
        for item in detections:
            if item.class_id not in PERSISTENT_PRESENCE_CLASS_IDS:
                continue
            if any(detections_match(previous, item) for previous in active):
                continue
            active.append(item)
        active_ids = {id(item) for item in active}
        background = [item for item in detections if id(item) not in active_ids]
        return cls(
            active,
            background,
            misses_to_stop=misses_to_stop,
            persistent_misses_to_stop=persistent_misses_to_stop,
            initial_confidence_threshold=initial_confidence_threshold,
            persistent_confidence_threshold=persistent_confidence_threshold,
        )

    def update(self, detections: Iterable[ObjectDetection], *, motion_active: bool) -> bool:
        all_detections = list(detections)
        detections = [
            item
            for item in all_detections
            if (
                item.class_id in PERSISTENT_PRESENCE_CLASS_IDS
                and item.confidence >= self.persistent_confidence_threshold
            )
            or (
                item.class_id not in PERSISTENT_PRESENCE_CLASS_IDS
                and motion_active
                and item.confidence >= self.initial_confidence_threshold
            )
        ]
        matched_indices = set()
        updated_active = []
        for previous in self.active:
            best_index = None
            best_iou = -1.0
            for index, current in enumerate(detections):
                if index in matched_indices or not detections_match(previous, current):
                    continue
                overlap = box_iou(previous.box, current.box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index is not None:
                matched_indices.add(best_index)
                current = detections[best_index]
                if (
                    previous.class_id in PERSISTENT_PRESENCE_CLASS_IDS
                    or detection_box_moved(previous, current)
                ):
                    updated_active.append(current)

        # A recording is about occupancy after it has been opened, not only
        # about the object that produced the original motion region.  Promote
        # every reliably detected person or animal even when it was initially
        # classified as background.  The initial threshold prevents a new weak
        # false positive from becoming a track; after promotion, normal
        # presence hysteresis retains it at the lower threshold.  Vehicles keep
        # the stricter movement and background checks below so a parked vehicle
        # cannot hold a file open.
        for index, current in enumerate(detections):
            if index in matched_indices:
                continue
            if current.class_id not in PERSISTENT_PRESENCE_CLASS_IDS:
                continue
            if current.confidence < self.initial_confidence_threshold:
                continue
            if any(detections_match(previous, current) for previous in updated_active):
                continue
            matched_indices.add(index)
            updated_active.append(current)

        if motion_active:
            for current in all_detections:
                if current.class_id in PERSISTENT_PRESENCE_CLASS_IDS:
                    continue
                if current.confidence < self.initial_confidence_threshold:
                    continue
                if any(current is matched for matched in updated_active):
                    continue
                if any(detections_match(previous, current) for previous in self.active):
                    continue
                if any(detections_match(background, current) for background in self.background):
                    continue
                updated_active.append(current)

        if updated_active:
            # A partial detection must not erase other known occupants.  Keep
            # unmatched living tracks as search anchors for later zoomed
            # checks.  They do not make this update successful by themselves:
            # when no current detection exists, the normal miss counter below
            # still advances and eventually stops the recording.
            retained_active = list(updated_active)
            for previous in self.active:
                if previous.class_id not in PERSISTENT_PRESENCE_CLASS_IDS:
                    continue
                if any(detections_match(previous, current) for current in retained_active):
                    continue
                retained_active.append(previous)
                if len(retained_active) >= YOLO_PRESENCE_MAX_ACTIVE_TRACKS:
                    break
            self.active = retained_active
            self.consecutive_misses = 0
            return True
        self.consecutive_misses += 1
        return False

    @property
    def effective_misses_to_stop(self) -> int:
        if any(item.class_id in PERSISTENT_PRESENCE_CLASS_IDS for item in self.active):
            return self.persistent_misses_to_stop
        return self.misses_to_stop

    @property
    def should_stop(self) -> bool:
        return self.consecutive_misses >= self.effective_misses_to_stop
