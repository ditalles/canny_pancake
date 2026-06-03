"""
Zicht-/betrouwbaarheidsbeoordeling (robuustheid bij weer & nacht).

Kern van een *betrouwbare* monitor: weten wanneer je iets NIET kunt zien.
Per frame berekenen we beeldkwaliteit-metrics en vatten die samen in een
zicht-score (0..1) en een niveau:

    GOOD     -> metingen worden vertrouwd
    DEGRADED -> metingen worden getoond maar gemarkeerd als minder zeker
    BLIND    -> zicht onbetrouwbaar; anomalie-alarmen worden ONDERDRUKT
                (eerlijk i.p.v. valse alarmen bij regen/sneeuw/nacht/mist)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import Config

HEALTH_GOOD = "GOOD"
HEALTH_DEGRADED = "DEGRADED"
HEALTH_BLIND = "BLIND"


@dataclass
class HealthReport:
    level: str
    visibility: float                 # 0..1 samengestelde zicht-score
    metrics: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True als metingen vertrouwd mogen worden (niet BLIND)."""
        return self.level != HEALTH_BLIND


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def assess_health(gray: np.ndarray,
                  depth_mm: Optional[np.ndarray],
                  feature_count: int,
                  cfg: Config) -> HealthReport:
    """Bereken de zicht-score uit beeldkwaliteit, depth-dekking en flow."""
    h, w = gray.shape[:2]
    reasons: List[str] = []

    # --- Helderheid (nacht/onderbelicht of verblinding/overbelicht) ---
    brightness = float(gray.mean()) / 255.0

    # --- Scherpte: variantie van de Laplaciaan. Laag = wazig (mist, regen
    #     op de lens, beslag, motion blur). Genormaliseerd met een schaal
    #     die ~scherp beeld op ~1.0 zet. ---
    focus_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    focus = _clamp01(focus_raw / 300.0)

    # --- Neerslag-index: regen/sneeuw geeft veel kleine high-contrast
    #     speckles. We vergelijken met een mediaan-geblurde versie en tellen
    #     de fractie sterk afwijkende pixels. ---
    blur = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, blur)
    precip = float((diff > 40).mean())

    # --- Depth-dekking in de centrale ROI ---
    depth_valid = 0.0
    if depth_mm is not None:
        y0, y1 = int(h * 0.30), int(h * 0.70)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        roi = depth_mm[y0:y1, x0:x1]
        if roi.size:
            depth_valid = float((roi > 0).mean())

    # --- Deelscores 0..1 (1 = goed) ---
    s_bright = 1.0 if cfg.min_brightness <= brightness <= cfg.max_brightness else 0.0
    s_focus = _clamp01(focus / max(cfg.min_focus, 1e-3))
    s_focus = min(1.0, s_focus)
    s_precip = _clamp01(1.0 - precip / max(cfg.max_precip, 1e-3))
    s_depth = _clamp01(depth_valid / max(cfg.min_depth_valid, 1e-3))
    s_depth = min(1.0, s_depth)
    s_feat = _clamp01(feature_count / max(cfg.min_features, 1))

    if not s_bright:
        reasons.append("belichting" + (" te donker" if brightness < cfg.min_brightness
                                       else " te fel"))
    if s_focus < 1.0:
        reasons.append("wazig/mist/regen op lens")
    if precip > cfg.max_precip:
        reasons.append("neerslag (regen/sneeuw)")
    if s_depth < 1.0:
        reasons.append("weinig geldige depth")
    if feature_count < cfg.min_features:
        reasons.append("te weinig optical-flow features")

    # --- Samengestelde zicht-score: zwakste schakels wegen zwaar
    #     (geometrisch gemiddelde -> 1 slechte metric trekt het omlaag). ---
    parts = [s_bright, s_focus, s_precip, s_depth, s_feat]
    parts = [max(p, 1e-3) for p in parts]
    visibility = float(np.exp(np.mean(np.log(parts))))

    if visibility < cfg.blind_visibility:
        level = HEALTH_BLIND
    elif visibility < cfg.degraded_visibility:
        level = HEALTH_DEGRADED
    else:
        level = HEALTH_GOOD
        reasons = []  # alles goed -> geen ruis in de redenen

    return HealthReport(
        level=level,
        visibility=visibility,
        metrics={
            "brightness": brightness,
            "focus": focus,
            "precip": precip,
            "depth_valid": depth_valid,
            "features": float(feature_count),
        },
        reasons=reasons,
    )
