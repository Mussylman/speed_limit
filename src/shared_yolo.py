"""
Shared YOLO: one model, one worker thread, N cameras.

Each camera gets its own ByteTrack state, swapped in/out before inference.
CameraYOLOProxy provides the same API as AsyncYOLO so main.py needs no changes.
"""

import copy
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

from async_yolo import YOLOResult, DetectionData


@dataclass
class SharedYOLOTask:
    """Task for shared YOLO queue."""
    frame: np.ndarray
    frame_idx: int
    camera_id: str
    submit_time: float


class SharedAsyncYOLO:
    """
    Single YOLO model serving multiple cameras sequentially.

    Architecture:
    - One input queue (all cameras submit here)
    - One worker thread dequeues, swaps ByteTrack state, runs inference
    - Result is routed to the submitting camera's output queue
    """

    def __init__(
        self,
        model,
        imgsz: int = 1280,
        classes: List[int] = None,
        half: bool = True,
        tracker: str = "bytetrack.yaml",
        max_queue_per_camera: int = 3,
        min_conf: float = 0.5,
        max_detect_size: int = 0,
    ):
        self.model = model
        self.imgsz = imgsz
        self.classes = classes or [2]
        self.half = half
        self.tracker = tracker
        self.min_conf = min_conf
        self.max_detect_size = max_detect_size

        self.input_queue: Queue[SharedYOLOTask] = Queue(maxsize=max_queue_per_camera * 3)
        self._camera_output_queues: Dict[str, Queue] = {}
        self._camera_trackers: Dict[str, Any] = {}
        self._last_camera_id: Optional[str] = None

        # Per-camera stats
        self._camera_stats: Dict[str, dict] = {}

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True,
                                              name="SharedYOLO")
        self.worker_thread.start()

        self.stats = {
            "submitted": 0,
            "processed": 0,
            "dropped": 0,
        }

    def register_camera(self, camera_id: str):
        """Register a camera: creates output queue and tracker slot."""
        self._camera_output_queues[camera_id] = Queue()
        self._camera_trackers[camera_id] = None  # populated after first inference
        self._camera_stats[camera_id] = {
            "submitted": 0,
            "processed": 0,
            "dropped": 0,
        }

    def _swap_tracker(self, camera_id: str):
        """Swap ByteTrack state for the given camera."""
        if camera_id == self._last_camera_id:
            return  # same camera as last call, no swap needed

        predictor = getattr(self.model, 'predictor', None)
        if predictor is None:
            # First inference hasn't happened yet, predictor will be created
            self._last_camera_id = camera_id
            return

        # Save current tracker state for previous camera
        if self._last_camera_id is not None and hasattr(predictor, 'trackers'):
            self._camera_trackers[self._last_camera_id] = copy.deepcopy(predictor.trackers)

        # Restore tracker for new camera
        saved = self._camera_trackers.get(camera_id)
        if saved is not None:
            predictor.trackers = copy.deepcopy(saved)
        else:
            # First call for this camera: delete trackers so model auto-reinits
            if hasattr(predictor, 'trackers'):
                delattr(predictor, 'trackers')

        self._last_camera_id = camera_id

    def _worker(self):
        """Single worker processing all cameras sequentially."""
        while self.running:
            try:
                task = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            if task is None:  # stop signal
                break

            queue_time_ms = (time.time() - task.submit_time) * 1000
            camera_id = task.camera_id

            t_start = time.time()
            orig_frame = task.frame
            h, w = orig_frame.shape[:2]

            # Pre-resize for detection
            detect_frame = orig_frame
            scale_back = 1.0
            if self.max_detect_size > 0 and h > self.max_detect_size:
                scale_back = h / self.max_detect_size
                new_h = self.max_detect_size
                new_w = int(w / scale_back)
                detect_frame = cv2.resize(orig_frame, (new_w, new_h),
                                          interpolation=cv2.INTER_AREA)

            try:
                # Swap tracker state for this camera
                self._swap_tracker(camera_id)

                results = self.model.track(
                    detect_frame,
                    imgsz=self.imgsz,
                    classes=self.classes,
                    persist=True,
                    verbose=False,
                    half=self.half,
                    tracker=self.tracker,
                )
                processing_time_ms = (time.time() - t_start) * 1000

                # Save tracker state after inference (for swap on next call)
                # (done lazily in _swap_tracker when camera changes)

                detections = []
                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes_obj = results[0].boxes
                    xyxy = boxes_obj.xyxy.cpu().numpy()
                    ids = boxes_obj.id.int().cpu().numpy()
                    confs = boxes_obj.conf.cpu().numpy()

                    for i in range(len(ids)):
                        x1 = xyxy[i][0] * scale_back
                        y1 = xyxy[i][1] * scale_back
                        x2 = xyxy[i][2] * scale_back
                        y2 = xyxy[i][3] * scale_back
                        obj_id = int(ids[i])
                        conf = float(confs[i])

                        if conf < self.min_conf:
                            continue

                        x1i, y1i = max(0, int(x1)), max(0, int(y1))
                        x2i, y2i = min(w, int(x2)), min(h, int(y2))

                        if x2i <= x1i or y2i <= y1i:
                            continue

                        crop = orig_frame[y1i:y2i, x1i:x2i].copy()
                        if crop.size == 0:
                            continue

                        cx = int((x1 + x2) / 2)
                        cy = int(y2)

                        detections.append(DetectionData(
                            obj_id=obj_id,
                            box=(x1i, y1i, x2i, y2i),
                            conf=conf,
                            crop=crop,
                            cx=cx,
                            cy=cy,
                        ))

                result = YOLOResult(
                    frame_idx=task.frame_idx,
                    detections=detections,
                    frame_shape=(h, w),
                    processing_time_ms=processing_time_ms,
                    queue_time_ms=queue_time_ms,
                )

            except Exception as e:
                print(f"\n[SharedYOLO] Error ({camera_id}): {e}")
                result = YOLOResult(
                    frame_idx=task.frame_idx,
                    detections=[],
                    frame_shape=(h, w),
                    processing_time_ms=(time.time() - t_start) * 1000,
                    queue_time_ms=queue_time_ms,
                )

            # Route result to camera's output queue
            out_q = self._camera_output_queues.get(camera_id)
            if out_q:
                out_q.put(result)

            self.stats["processed"] += 1
            cam_stats = self._camera_stats.get(camera_id)
            if cam_stats:
                cam_stats["processed"] += 1

    def stop(self):
        """Stop the shared worker. Call after all camera threads have joined."""
        self.running = False
        try:
            self.input_queue.put_nowait(None)
        except Exception:
            pass
        self.worker_thread.join(timeout=5.0)

        total = self.stats
        submitted = total["submitted"]
        drop_pct = (total["dropped"] / submitted * 100) if submitted > 0 else 0
        print(f"\n[SharedYOLO] Summary:")
        print(f"  Total: submit={submitted} done={total['processed']} "
              f"drop={total['dropped']} ({drop_pct:.1f}%)")
        for cam_id, cs in self._camera_stats.items():
            cs_sub = cs["submitted"]
            cs_drop_pct = (cs["dropped"] / cs_sub * 100) if cs_sub > 0 else 0
            print(f"  {cam_id}: submit={cs_sub} done={cs['processed']} "
                  f"drop={cs['dropped']} ({cs_drop_pct:.1f}%)")


class CameraYOLOProxy:
    """
    Drop-in replacement for AsyncYOLO.

    Routes submit/get_results through SharedAsyncYOLO's queues,
    tagged with camera_id. Same API so main.py needs no changes.
    """

    def __init__(self, shared: SharedAsyncYOLO, camera_id: str):
        self.shared = shared
        self.camera_id = camera_id
        shared.register_camera(camera_id)

        # main.py uses async_yolo.input_queue.qsize() for logging
        self.input_queue = shared.input_queue

    def submit(self, frame: np.ndarray, frame_idx: int, blocking: bool = False) -> bool:
        task = SharedYOLOTask(
            frame=frame.copy(),
            frame_idx=frame_idx,
            camera_id=self.camera_id,
            submit_time=time.time(),
        )

        cam_stats = self.shared._camera_stats.get(self.camera_id, self.shared.stats)

        if blocking:
            self.shared.input_queue.put(task)
            self.shared.stats["submitted"] += 1
            cam_stats["submitted"] += 1
            return True

        if self.shared.input_queue.full():
            self.shared.stats["dropped"] += 1
            cam_stats["dropped"] += 1
            return False

        try:
            self.shared.input_queue.put_nowait(task)
            self.shared.stats["submitted"] += 1
            cam_stats["submitted"] += 1
            return True
        except Exception:
            self.shared.stats["dropped"] += 1
            cam_stats["dropped"] += 1
            return False

    def get_results(self, blocking: bool = False) -> List[YOLOResult]:
        out_q = self.shared._camera_output_queues[self.camera_id]
        results = []
        if blocking:
            try:
                result = out_q.get(timeout=5.0)
                results.append(result)
            except Empty:
                return results
        while True:
            try:
                result = out_q.get_nowait()
                results.append(result)
            except Empty:
                break
        return results

    def get_latest_result(self) -> Optional[YOLOResult]:
        out_q = self.shared._camera_output_queues[self.camera_id]
        latest = None
        while True:
            try:
                result = out_q.get_nowait()
                latest = result
            except Empty:
                break
        return latest

    def stop(self):
        """No-op: shared YOLO outlives individual camera threads."""
        pass

    def get_stats(self) -> dict:
        cam_stats = self.shared._camera_stats.get(self.camera_id, {})
        return {
            "submitted": cam_stats.get("submitted", 0),
            "processed": cam_stats.get("processed", 0),
            "dropped": cam_stats.get("dropped", 0),
            "input_queue_size": self.shared.input_queue.qsize(),
            "output_queue_size": self.shared._camera_output_queues[self.camera_id].qsize(),
            "gpu_lock_wait_ms": 0.0,
            "gpu_lock_count": 0,
            "gpu_lock_avg_ms": 0.0,
        }
