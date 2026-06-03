"""
Perceptie: odometer (snelheid/afstand), catenary-gecorrigeerde diktemeting
en anomalie-/catenary-beoordeling. Pure metingen, geen I/O.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .health import HealthReport
from .sources import Intrinsics

# Status-codes (operator-georienteerd, GEEN machinebesturing)
STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_CRIT = "CRIT"
STATUS_UNKNOWN = "UNKNOWN"   # zicht onbetrouwbaar -> meting niet te vertrouwen


# ======================================================================
# Wiskunde: pixel <-> millimeter via pinhole-model
# ======================================================================
def pixels_to_mm(pixel_size: float, z_m: float, focal_px: float) -> float:
    """
    Pinhole: een object met reele grootte W op afstand Z projecteert op
        p = focal_px * W / Z  pixels.
    Inverse (pixels -> meters):  W = p * Z / focal_px.  (*1000 -> mm)

    Doordat de LIVE Z in de formule zit is de meting AFSTAND-ONAFHANKELIJK:
    hangt de kabel dichterbij, dan worden pixels met een kleinere factor naar
    mm geschaald -> dit is de catenary-correctie.
    """
    if focal_px <= 0 or z_m <= 0:
        return 0.0
    return (pixel_size * z_m / focal_px) * 1000.0


def median_depth_at(depth_mm: np.ndarray, x0, x1, y0, y1) -> float:
    """Robuuste mediaan-Z (m) over een ROI; negeert ongeldige (0) pixels."""
    h, w = depth_mm.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = depth_mm[y0:y1, x0:x1]
    valid = roi[roi > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) / 1000.0


# ======================================================================
# Diktemeting
# ======================================================================
@dataclass
class ThicknessResult:
    thickness_px: float
    y_top: float
    y_bottom: float
    y_center_frac: float


def cable_depth_m(depth_mm: np.ndarray, meas: "ThicknessResult",
                  cfg: Config) -> float:
    """
    Mediaan-Z (m) gemeten OP de kabel zelf (rijen tussen de gedetecteerde
    boven/onder-rand, kolommen in de meet-strook). Dit is de juiste catenary-
    afstand en de Z die in de pixel->mm correctie hoort -> niet de achtergrond.
    """
    h, w = depth_mm.shape[:2]
    c0, c1 = int(w * cfg.roi_col_lo), int(w * cfg.roi_col_hi)
    return median_depth_at(depth_mm, c0, c1, int(meas.y_top), int(meas.y_bottom))


def measure_thickness_px(gray: np.ndarray, cfg: Config) -> Optional[ThicknessResult]:
    """
    Verticale kabeldikte in pixels via Canny-randen, per kolom in een centrale
    strook. Mediaan over kolommen -> robuust tegen ruis. Geeft ook het
    verticale midden (voor de Y/catenary-check) terug.
    """
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    c0, c1 = int(w * cfg.roi_col_lo), int(w * cfg.roi_col_hi)

    spans, tops, bots = [], [], []
    for x in range(c0, c1, 4):
        ys = np.where(edges[:, x] > 0)[0]
        if ys.size >= 2:
            yt, yb = ys[0], ys[-1]
            span = yb - yt
            if 5 < span < h * 0.9:        # filter ruis en bijna-volbeeld
                spans.append(span); tops.append(yt); bots.append(yb)

    if len(spans) < 5:
        return None
    yt = float(np.median(tops)); yb = float(np.median(bots))
    return ThicknessResult(
        thickness_px=float(np.median(spans)),
        y_top=yt, y_bottom=yb,
        y_center_frac=((yt + yb) / 2.0) / h,
    )


# ======================================================================
# Odometer
# ======================================================================
class Odometer:
    """
    Snelheid (cm/s) en afgelegde afstand (m) uit sparse optical flow + Z.

    Per feature-ID die tussen twee frames bestaat kennen we de pixel-
    verplaatsing dp; via het pinhole-model + live Z volgt de reele
    verplaatsing dX = dp * Z / fx. Mediaan over features -> robuust.
    """

    def __init__(self, intr: Intrinsics, cfg: Config):
        self.intr = intr
        self.cfg = cfg
        self.prev = {}
        self.prev_ts: Optional[float] = None
        self.total_distance_m = 0.0
        self.speed_hist = deque(maxlen=cfg.smooth_window)

    def update(self, features: List[Tuple[int, float, float]],
               ts: float, z_m: float, trust: bool = True) -> float:
        cur = {fid: (x, y) for fid, x, y in features}

        if self.prev_ts is None or z_m <= 0:
            self.prev, self.prev_ts = cur, ts
            return self.speed()

        dt = ts - self.prev_ts
        if dt <= 1e-4:
            return self.speed()

        disp = []
        for fid, (x, y) in cur.items():
            if fid in self.prev:
                px, py = self.prev[fid]
                dp = math.hypot(x - px, y - py)
                disp.append(dp * z_m / self.intr.fx)   # pixels -> meters

        # Bij onbetrouwbaar zicht wel de toestand bijwerken, maar afstand
        # NIET integreren (anders telt ruis mee in de meterstand).
        if disp and trust:
            med = float(np.median(disp))
            self.speed_hist.append((med / dt) * 100.0)  # m/s -> cm/s
            self.total_distance_m += med

        self.prev, self.prev_ts = cur, ts
        return self.speed()

    def speed(self) -> float:
        return float(np.median(self.speed_hist)) if self.speed_hist else 0.0


# ======================================================================
# Anomalie-/catenary-beoordeling
# ======================================================================
@dataclass
class Assessment:
    status: str
    reasons: List[str]
    deviation: float          # relatieve dikte-afwijking


def assess(diameter_mm: float, baseline_mm: float, z_m: float,
           y_center_frac: float, health: HealthReport,
           cfg: Config) -> Assessment:
    """
    Combineer dikte-afwijking + catenary (Z/Y) tot een operator-status.
    Bij BLIND zicht: status UNKNOWN en GEEN anomalie-alarm (eerlijk i.p.v.
    vals). Catenary blijft wel gemeld als de depth bruikbaar is.
    """
    reasons: List[str] = []
    deviation = 0.0

    if not health.usable:
        return Assessment(STATUS_UNKNOWN,
                          ["Zicht onbetrouwbaar: " + ", ".join(health.reasons)
                           or "Zicht onbetrouwbaar"],
                          0.0)

    status = STATUS_OK

    # --- Catenary: Z buiten venster ---
    if z_m > 0:
        if z_m < cfg.z_min_m:
            status = STATUS_CRIT
            reasons.append(f"Kabel te SLAP/dichtbij (Z={z_m:.2f}m)")
        elif z_m > cfg.z_max_m:
            status = STATUS_CRIT
            reasons.append(f"Kabel te STRAK/ver (Z={z_m:.2f}m)")

    # --- Y-positie: uit geleiderrol ---
    if not (cfg.y_center_min <= y_center_frac <= cfg.y_center_max):
        status = STATUS_CRIT
        reasons.append(f"Kabel-as buiten band (Y={y_center_frac:.2f})")

    # --- Dikte-afwijking ---
    if baseline_mm > 0 and diameter_mm > 0:
        deviation = abs(diameter_mm - baseline_mm) / baseline_mm
        if deviation > cfg.crit_deviation:
            status = STATUS_CRIT
            reasons.append(f"Dikte {deviation*100:.0f}% afw. (birdcage?) "
                           f"{diameter_mm:.0f}mm vs {baseline_mm:.0f}mm")
        elif deviation > cfg.warn_deviation and status != STATUS_CRIT:
            status = STATUS_WARN
            reasons.append(f"Dikte {deviation*100:.0f}% afw. (tape?)")

    if not reasons:
        reasons.append("Binnen tolerantie")
    return Assessment(status, reasons, deviation)
