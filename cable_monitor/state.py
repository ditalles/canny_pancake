"""
Gedeelde, thread-safe monitor-toestand.

De perceptie-engine draait in een eigen thread en schrijft hierheen; het
Flask-dashboard leest hieruit. Eenvoudige lock-beschermde snapshot.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .config import Config


class MonitorState:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()

        # Laatste meetwaarden
        self.distance_m = 0.0
        self.speed_cms = 0.0
        self.diameter_mm = 0.0
        self.baseline_mm = cfg.nominal_diameter_mm
        self.z_m = 0.0
        self.y_center = 0.5
        self.deviation = 0.0

        # Status
        self.status = "OK"
        self.reasons = ["Opstarten..."]
        self.health = "GOOD"
        self.visibility = 1.0
        self.metrics = {}
        self.calibrating = cfg.autocalibrate_baseline

        # Tellers
        self.updated_at = 0.0
        self.fps = 0.0

        # Historie voor de grafiek
        self.history = deque(maxlen=cfg.history_len)

        # Laatste geannoteerde JPEG (voor de MJPEG-stream)
        self._jpeg: Optional[bytes] = None

    # ------------------------------------------------------------------
    def update(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)
            self.updated_at = time.time()
            self.history.append({
                "t": self.updated_at,
                "diameter": self.diameter_mm,
                "baseline": self.baseline_mm,
                "z": self.z_m,
                "speed": self.speed_cms,
                "visibility": self.visibility,
                "status": self.status,
            })

    def set_frame(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            age = time.time() - self.updated_at if self.updated_at else None
            return {
                "distance_m": round(self.distance_m, 2),
                "speed_cms": round(self.speed_cms, 1),
                "diameter_mm": round(self.diameter_mm, 1),
                "baseline_mm": round(self.baseline_mm, 1),
                "z_m": round(self.z_m, 3),
                "y_center": round(self.y_center, 3),
                "deviation_pct": round(self.deviation * 100, 1),
                "status": self.status,
                "reasons": self.reasons,
                "health": self.health,
                "visibility": round(self.visibility, 3),
                "metrics": {k: round(v, 3) for k, v in self.metrics.items()},
                "calibrating": self.calibrating,
                "fps": round(self.fps, 1),
                "stale_s": round(age, 1) if age is not None else None,
                "history": list(self.history),
            }
