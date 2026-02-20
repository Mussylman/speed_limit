# cross_camera_speed.py
# Cross-camera average speed: same plate on 2+ cameras → speed = distance / time

import json
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class Sighting:
    camera_id: str
    timestamp: float
    frame_idx: int


@dataclass
class CrossSpeedResult:
    plate_text: str
    camera_from: str
    camera_to: str
    distance_m: float
    time_sec: float
    speed_kmh: float
    over_limit: bool
    timestamp: float  # when computed


class CrossCameraTracker:
    """
    Aggregates plate sightings from multiple cameras.
    When the same plate is seen on 2+ cameras with known distance,
    calculates average speed = distance / time_difference.

    Runs as a daemon thread in the main process, reading from
    a multiprocessing.Queue fed by camera processes.
    """

    TTL_SEC = 300  # 5 min — max time to keep sightings

    def __init__(self, distances_config: list, speed_limit: int = 70,
                 output_dir: str = "output"):
        self.speed_limit = speed_limit
        self.output_dir = output_dir

        # Build distance lookup: {("CamA","CamB"): meters}
        self._distances: Dict[Tuple[str, str], float] = {}
        self._cameras_ordered: List[str] = []  # ordered sequence for summing
        self._build_distances(distances_config)

        # {plate_text: [Sighting, ...]} — sorted by timestamp
        self.sightings: Dict[str, List[Sighting]] = defaultdict(list)

        # Computed results
        self.results: List[CrossSpeedResult] = []

        # Dedup: avoid computing same pair twice
        # key = (plate_text, camera_from, camera_to)
        self._computed: set = set()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._queue = None

        # Stats
        self.stats = {"events_received": 0, "speeds_computed": 0,
                      "violations": 0, "cleanups": 0}

    def _build_distances(self, distances_config: list):
        """Parse distance pairs and build lookup + ordered camera list."""
        seen_cameras = []
        for entry in distances_config:
            cams = entry["cameras"]
            dist = float(entry["distance"])
            a, b = cams[0], cams[1]
            self._distances[(a, b)] = dist
            self._distances[(b, a)] = dist
            for c in (a, b):
                if c not in seen_cameras:
                    seen_cameras.append(c)
        self._cameras_ordered = seen_cameras

        # Auto-sum indirect distances (A→C = A→B + B→C)
        changed = True
        while changed:
            changed = False
            for i, ca in enumerate(self._cameras_ordered):
                for j, cc in enumerate(self._cameras_ordered):
                    if i >= j:
                        continue
                    if (ca, cc) in self._distances:
                        continue
                    # Try to find intermediate camera
                    for cb in self._cameras_ordered:
                        if cb == ca or cb == cc:
                            continue
                        d1 = self._distances.get((ca, cb))
                        d2 = self._distances.get((cb, cc))
                        if d1 is not None and d2 is not None:
                            total = d1 + d2
                            self._distances[(ca, cc)] = total
                            self._distances[(cc, ca)] = total
                            changed = True
                            break

    def start(self, queue):
        """Start the reader thread."""
        self._queue = queue
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="CrossCameraTracker")
        self._thread.start()

    def _run(self):
        """Main loop: read events from queue, process them."""
        last_cleanup = time.time()
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=1.0)
            except Exception:
                # Timeout or empty — check cleanup
                if time.time() - last_cleanup > 30:
                    self._cleanup()
                    last_cleanup = time.time()
                continue

            self._process_event(event)

            if time.time() - last_cleanup > 30:
                self._cleanup()
                last_cleanup = time.time()

    def _process_event(self, event: dict):
        """Handle a single plate sighting event."""
        plate = event["plate_text"]
        cam = event["camera_id"]
        ts = event["timestamp"]
        fidx = event["frame_idx"]

        with self._lock:
            self.stats["events_received"] += 1

            sighting = Sighting(camera_id=cam, timestamp=ts, frame_idx=fidx)
            self.sightings[plate].append(sighting)

            # Check all pairs for this plate
            plate_sightings = self.sightings[plate]
            for i, s1 in enumerate(plate_sightings):
                for s2 in plate_sightings[i + 1:]:
                    if s1.camera_id == s2.camera_id:
                        continue

                    pair_key = (plate, s1.camera_id, s2.camera_id)
                    reverse_key = (plate, s2.camera_id, s1.camera_id)
                    if pair_key in self._computed or reverse_key in self._computed:
                        continue

                    dist = self._distances.get((s1.camera_id, s2.camera_id))
                    if dist is None:
                        continue

                    # Ensure correct order (earlier → later)
                    if s1.timestamp <= s2.timestamp:
                        cam_from, cam_to = s1.camera_id, s2.camera_id
                        dt = s2.timestamp - s1.timestamp
                    else:
                        cam_from, cam_to = s2.camera_id, s1.camera_id
                        dt = s1.timestamp - s2.timestamp

                    if dt < 1.0:
                        continue  # too fast, likely same camera or error

                    speed_ms = dist / dt
                    speed_kmh = speed_ms * 3.6
                    over_limit = speed_kmh > self.speed_limit

                    result = CrossSpeedResult(
                        plate_text=plate,
                        camera_from=cam_from,
                        camera_to=cam_to,
                        distance_m=dist,
                        time_sec=round(dt, 1),
                        speed_kmh=round(speed_kmh, 1),
                        over_limit=over_limit,
                        timestamp=time.time(),
                    )
                    self.results.append(result)
                    self._computed.add(pair_key)
                    self.stats["speeds_computed"] += 1
                    if over_limit:
                        self.stats["violations"] += 1

                    marker = " !!!" if over_limit else ""
                    print(f"\n[CROSS-CAM] {plate} | {cam_from} -> {cam_to} | "
                          f"{dist:.0f}m / {dt:.1f}s = {speed_kmh:.1f} km/h{marker}")

    def _cleanup(self):
        """Remove sightings older than TTL."""
        now = time.time()
        cutoff = now - self.TTL_SEC
        with self._lock:
            to_remove = []
            for plate, slist in self.sightings.items():
                slist[:] = [s for s in slist if s.timestamp > cutoff]
                if not slist:
                    to_remove.append(plate)
            for plate in to_remove:
                del self.sightings[plate]
                # Also clean computed keys for this plate
                self._computed = {k for k in self._computed if k[0] != plate}
            if to_remove:
                self.stats["cleanups"] += len(to_remove)

    def stop(self):
        """Signal thread to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def finalize(self):
        """Stop thread and save results."""
        self.stop()
        # Drain remaining events from queue
        if self._queue:
            while True:
                try:
                    event = self._queue.get_nowait()
                    self._process_event(event)
                except Exception:
                    break

        self._save_results()

    def _save_results(self):
        """Save results to JSON file."""
        if not self.results:
            print("[CROSS-CAM] No cross-camera speed results.")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "cross_camera_speeds.json")
        data = {
            "speed_limit": self.speed_limit,
            "results": [asdict(r) for r in self.results],
            "stats": self.stats,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[CROSS-CAM] Saved {len(self.results)} results → {path}")

    def print_summary(self):
        """Print summary to console."""
        print(f"\n{'='*60}")
        print(f"[CROSS-CAM] Summary: {self.stats['events_received']} events, "
              f"{self.stats['speeds_computed']} speeds, "
              f"{self.stats['violations']} violations")
        if self.results:
            for r in self.results:
                marker = " !!!" if r.over_limit else ""
                print(f"  {r.plate_text}: {r.camera_from} -> {r.camera_to} | "
                      f"{r.distance_m:.0f}m / {r.time_sec:.1f}s = "
                      f"{r.speed_kmh:.1f} km/h{marker}")
        print(f"{'='*60}")
