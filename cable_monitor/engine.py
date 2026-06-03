"""
Monitor-engine: knoopt bron -> perceptie -> health -> toestand aan elkaar.

MONITORING ONLY. Geen machinebesturing. Bij een afwijking roept de engine
`raise_operator_alert()` aan: dat logt/telemetreert puur, zodat de OPERATOR
via het dashboard kan beslissen in te grijpen.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import cv2
import numpy as np

from .config import Config
from .health import assess_health, HEALTH_GOOD
from .perception import (Assessment, Odometer, assess, cable_depth_m,
                         measure_thickness_px, median_depth_at, pixels_to_mm,
                         STATUS_CRIT, STATUS_WARN)
from .sources import FrameSource
from .state import MonitorState

# Status -> kleur (BGR) en label
_COLOR = {
    "OK": (0, 200, 0),
    "WARN": (0, 165, 255),
    "CRIT": (0, 0, 255),
    "UNKNOWN": (160, 160, 160),
}
_LABEL = {
    "OK": "OK",
    "WARN": "WAARSCHUWING",
    "CRIT": "KRITIEK",
    "UNKNOWN": "ZICHT ONBETROUWBAAR",
}


# ----------------------------------------------------------------------
# Operator-koppelingen (advies, GEEN besturing)
# ----------------------------------------------------------------------
def raise_operator_alert(level: str, reasons) -> None:
    """
    PLACEHOLDER advies-signaal naar de operator.

    Bewust GEEN actuator/noodstop: deze monitor stuurt geen machines aan.
    Hier zou je bijv. een dashboard-melding, sirene, of e-mail/SMS triggeren
    zodat een mens beslist. In productie evt. een NIET-veilig digitaal signaal
    naar de leitstand; de daadwerkelijke stop loopt via de gecertificeerde
    besturing.
    """
    print(f"[OPERATOR-ALERT/{level}] {', '.join(reasons)}")


def send_mqtt_update(payload: dict, cfg: Config) -> None:
    """PLACEHOLDER telemetrie naar een MQTT-dashboard (advies, geen besturing)."""
    if not cfg.mqtt_enabled:
        return
    # client.publish("cable/catenary", json.dumps(payload))
    print(f"[MQTT] {json.dumps(payload)}")


# ======================================================================
class MonitorEngine:
    def __init__(self, source: FrameSource, state: MonitorState, cfg: Config):
        self.source = source
        self.state = state
        self.cfg = cfg
        self.odometer: Optional[Odometer] = None
        self.baseline_mm = cfg.nominal_diameter_mm
        self._baseline_samples = []
        self._calibrating = cfg.autocalibrate_baseline
        self._last_mqtt = 0.0
        self._last_alert_status = "OK"
        self._fps_t = time.time()
        self._fps_n = 0
        self._running = False

    # ------------------------------------------------------------------
    def _fps(self) -> float:
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            fps = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0
            return fps
        return self.state.fps

    # ------------------------------------------------------------------
    def step(self) -> Optional[np.ndarray]:
        """Verwerk een frame; geeft het geannoteerde beeld terug (of None)."""
        bundle = self.source.read()
        if bundle is None:
            return None

        cfg = self.cfg
        frame = bundle.rgb
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if self.odometer is None:
            self.odometer = Odometer(bundle.intrinsics, cfg)

        # --- Health / zicht beoordelen (robuustheid) ---
        health = assess_health(gray, bundle.depth_mm, len(bundle.features), cfg)
        trust = health.usable

        # --- Diktemeting eerst: bepaalt waar de kabel zit ---
        diameter_mm = 0.0
        y_center = 0.5
        meas = measure_thickness_px(gray, cfg)

        # --- Catenary-afstand Z: bij voorkeur OP de kabel gemeten, anders
        #     terugvallen op een centrale ROI. ---
        z_m = 0.0
        if bundle.depth_mm is not None:
            if meas is not None:
                z_m = cable_depth_m(bundle.depth_mm, meas, cfg)
            if z_m <= 0:
                z_m = median_depth_at(bundle.depth_mm,
                                      int(w * 0.35), int(w * 0.65),
                                      int(h * 0.35), int(h * 0.65))

        # --- Odometer ---
        speed = self.odometer.update(bundle.features, bundle.timestamp, z_m, trust)
        distance = self.odometer.total_distance_m

        # --- Diktemeting (catenary-gecorrigeerd) ---
        if meas is not None and z_m > 0:
            y_center = meas.y_center_frac
            # mm = px * Z / fy ; fy = verticale focal length
            diameter_mm = pixels_to_mm(meas.thickness_px, z_m, bundle.intrinsics.fy)

            # Baseline alleen kalibreren bij GOED zicht (schone referentie).
            if self._calibrating and trust and health.level == HEALTH_GOOD \
                    and diameter_mm > 0:
                self._baseline_samples.append(diameter_mm)
                if len(self._baseline_samples) >= cfg.autocalibrate_frames:
                    self.baseline_mm = float(np.median(self._baseline_samples))
                    self._calibrating = False
                    print(f"[CALIB] Baseline = {self.baseline_mm:.1f} mm")

        # --- Beoordeling ---
        if self._calibrating:
            a = Assessment("OK", ["Baseline kalibreren..."], 0.0)
        else:
            a = assess(diameter_mm, self.baseline_mm, z_m, y_center, health, cfg)

        # --- Advies-signaal bij nieuwe WARN/CRIT (geen spam) ---
        if a.status in (STATUS_WARN, STATUS_CRIT) and a.status != self._last_alert_status:
            raise_operator_alert(a.status, a.reasons)
        self._last_alert_status = a.status

        # --- Toestand bijwerken ---
        fps = self._fps()
        self.state.update(
            distance_m=distance, speed_cms=speed, diameter_mm=diameter_mm,
            baseline_mm=self.baseline_mm, z_m=z_m, y_center=y_center,
            deviation=a.deviation, status=a.status, reasons=a.reasons,
            health=health.level, visibility=health.visibility,
            metrics=health.metrics, calibrating=self._calibrating, fps=fps,
        )

        # --- Telemetrie (advies) ---
        now = time.time()
        if now - self._last_mqtt > cfg.mqtt_interval_s:
            send_mqtt_update(self.state.snapshot(), cfg)
            self._last_mqtt = now

        # --- Annoteren + naar de dashboard-videostream ---
        annotated = self._annotate(frame, meas, a, health)
        self.state.set_frame(annotated)
        return annotated

    # ------------------------------------------------------------------
    def _annotate(self, frame, meas, a: Assessment, health) -> np.ndarray:
        h, w = frame.shape[:2]
        color = _COLOR.get(a.status, (160, 160, 160))
        s = self.state

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 96), (30, 30, 30), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        def put(txt, y):
            cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        put(f"Afstand : {s.distance_m:7.2f} m", 22)
        put(f"Snelheid: {s.speed_cms:6.1f} cm/s", 44)
        put(f"Dikte   : {s.diameter_mm:6.1f} mm (basis {s.baseline_mm:.0f})", 66)
        put(f"Z={s.z_m:.2f}m  zicht={health.visibility:.0%} [{health.level}]", 88)

        cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
        cv2.circle(frame, (18, h - 15), 9, color, -1)
        cv2.putText(frame, f"{_LABEL.get(a.status)}: {'; '.join(a.reasons)}",
                    (36, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    cv2.LINE_AA)

        if meas is not None:
            yt, yb = int(meas.y_top), int(meas.y_bottom)
            x0 = int(w * self.cfg.roi_col_lo); x1 = int(w * self.cfg.roi_col_hi)
            cv2.line(frame, (x0, yt), (x1, yt), color, 1)
            cv2.line(frame, (x0, yb), (x1, yb), color, 1)

        cv2.rectangle(frame, (1, 1), (w - 2, h - 2), color, 3)
        return frame

    # ------------------------------------------------------------------
    def run(self, show_window: bool = False) -> None:
        """Blokkkerende loop. `show_window` opent een cv2-venster (lokaal)."""
        self._running = True
        with self.source:
            while self._running:
                annotated = self.step()
                if annotated is None:
                    time.sleep(0.002)
                else:
                    if show_window:
                        cv2.imshow("Catenary Monitor (q=stop)", annotated)
                if show_window and cv2.waitKey(1) == ord("q"):
                    break
        if show_window:
            cv2.destroyAllWindows()

    def stop(self) -> None:
        self._running = False
