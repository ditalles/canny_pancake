#!/usr/bin/env python3
"""
Cable Inspection PoC  -  OAK-D Lite FF + DepthAI
================================================

Industriele kabel-inspectie Proof of Concept.

Een zware, ronde maritieme kabel (7 km) loopt met ~10 m/min (~16.6 cm/s)
door een geleiderrol langs de OAK-D Lite. Op EEN enkele DepthAI-pijplijn
draaien drie functies parallel en real-time:

  1. Odometer & Snelheid   -> Optical Flow (FeatureTracker) + StereoDepth (Z)
  2. Dynamische Diktemeting -> verticale randdetectie + catenary (Z) correctie
  3. Anomaly / Catenary Alarm -> dikte-afwijking >15% of Z/Y buiten range

De pijplijn is bewust zo opgezet dat alleen RGB + depth + sparse features
over de USB-bus van de MacBook Air gaan (geen volle mono-streams), zodat
de USB-bandbreedte beheersbaar blijft.

Run:
    python cable_inspection_poc.py

Afsluiten:  toets 'q' in het OpenCV-venster.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import depthai as dai
except ImportError as exc:  # pragma: no cover - hardware/SDK afhankelijk
    raise SystemExit(
        "depthai is niet geinstalleerd. Installeer met:\n"
        "    pip install depthai opencv-python numpy"
    ) from exc


# ======================================================================
# CONFIGURATIE
# ======================================================================
@dataclass
class Config:
    # --- Camera / pijplijn -------------------------------------------
    rgb_width: int = 640            # preview/processing resolutie (breed)
    rgb_height: int = 400           # preview/processing resolutie (hoog)
    fps: int = 30

    # --- Kabel / fysiek ----------------------------------------------
    nominal_diameter_mm: float = 80.0   # verwachte basis-diameter van de kabel
    nominal_speed_cms: float = 16.6     # verwachte lijnsnelheid (~10 m/min)

    # --- Catenary (doorhang) veiligheidsvenster ----------------------
    # De kabel hoort op een bepaalde afstand van de camera te hangen.
    # Te dichtbij = te slap (doorhang), te ver = te strak getrokken.
    z_min_m: float = 0.40           # minimaal toegestane afstand camera->kabel
    z_max_m: float = 1.20           # maximaal toegestane afstand camera->kabel

    # Verticale positie (Y) van de kabel-as in beeld, als fractie 0..1.
    # Buiten deze band hangt de kabel scheef / uit de geleiderrol.
    y_center_min: float = 0.25
    y_center_max: float = 0.75

    # --- Alarm-drempels ----------------------------------------------
    warn_deviation: float = 0.10    # >10% afwijking -> ORANJE (tape-reparatie)
    crit_deviation: float = 0.15    # >15% afwijking -> ROOD (birdcage / kritiek)

    # --- Randdetectie -------------------------------------------------
    canny_low: int = 60
    canny_high: int = 180

    # --- Filtering (ruisonderdrukking op metingen) -------------------
    smooth_window: int = 8          # voortschrijdend gemiddelde venster

    # --- Auto-kalibratie van de basisdikte ---------------------------
    autocalibrate_baseline: bool = True
    autocalibrate_frames: int = 60  # eerste N goede frames -> baseline


CFG = Config()


# ======================================================================
# INDUSTRIELE KOPPELINGEN  (placeholders)
# ======================================================================
def trigger_emergency_stop(reason: str) -> None:
    """
    PLACEHOLDER voor de noodstop.

    In productie hangt hier de aansturing van een industrieel relais /
    safety-PLC (bv. via GPIO, Modbus of een digitale veiligheidsuitgang)
    die de lier/geleiderrol stopt.
    """
    print(f"[EMERGENCY-STOP] >>> {reason}")


def send_mqtt_update(payload: dict) -> None:
    """
    PLACEHOLDER voor het MQTT-dashboard.

    In productie publiceer je hier de telemetrie naar een broker, bv.:
        client.publish("cable/inspection", json.dumps(payload))
    Hier alleen een lichte console-log zodat de PoC zelfstandig draait.
    """
    # Bewust low-frequency loggen om de console niet te overspoelen.
    print(
        f"[MQTT] dist={payload['distance_m']:.1f}m "
        f"v={payload['speed_cms']:.1f}cm/s "
        f"d={payload['diameter_mm']:.1f}mm "
        f"status={payload['status']}"
    )


# ======================================================================
# 1. DEPTHAI-PIJPLIJN
# ======================================================================
def build_pipeline(cfg: Config) -> dai.Pipeline:
    """
    Bouwt EEN pijplijn met:
      - ColorCamera (CAM_A / RGB)        -> preview-stream
      - 2x MonoCamera (CAM_B / CAM_C)    -> StereoDepth
      - StereoDepth uitgelijnd op CAM_A  -> depth-map in RGB-perspectief
      - FeatureTracker op de RGB-feed    -> sparse optical flow (odometer)

    Door StereoDepth op CAM_A uit te lijnen valt elke RGB-pixel direct
    samen met een depth-pixel: handig om bij een rand-pixel meteen de
    Z-afstand op te zoeken. We sturen alleen RGB + depth + features over
    USB, niet de rauwe mono-beelden -> spaart bandbreedte.
    """
    pipeline = dai.Pipeline()

    # ---- RGB kleurcamera (CAM_A) ------------------------------------
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(cfg.rgb_width, cfg.rgb_height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(cfg.fps)

    # ---- Mono camera's voor stereo ----------------------------------
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    for mono, sock in (
        (mono_left, dai.CameraBoardSocket.CAM_B),
        (mono_right, dai.CameraBoardSocket.CAM_C),
    ):
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono.setBoardSocket(sock)
        mono.setFps(cfg.fps)

    # ---- StereoDepth, uitgelijnd op RGB -----------------------------
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    # Depth in het RGB-camera coordinatenstelsel -> 1:1 met preview-pixels.
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # ---- FeatureTracker op de RGB-feed (optical flow / odometer) ----
    feature_tracker = pipeline.create(dai.node.FeatureTracker)
    # Cap het aantal corners: genoeg voor robuuste flow, licht voor USB.
    feature_tracker.initialConfig.setNumTargetFeatures(256)
    feature_tracker.initialConfig.setMotionEstimator(True)
    cam_rgb.video.link(feature_tracker.inputImage)

    # ---- XLink uitgangen naar de host -------------------------------
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    xout_feat = pipeline.create(dai.node.XLinkOut)
    xout_feat.setStreamName("features")
    feature_tracker.outputFeatures.link(xout_feat.input)

    return pipeline


# ======================================================================
# KALIBRATIE  -  focal length dynamisch uit de OAK-D Lite chip
# ======================================================================
@dataclass
class Intrinsics:
    fx: float       # focal length in pixels (X)
    fy: float       # focal length in pixels (Y)
    cx: float
    cy: float
    width: int
    height: int


def read_intrinsics(device: dai.Device, cfg: Config) -> Intrinsics:
    """
    Haalt de fabrieks-kalibratie (intrinsics) live uit de EEPROM van de chip.

    Elke OAK-D wordt in de fabriek gekalibreerd; die data zit in de EEPROM
    op de camera zelf. We schalen de intrinsics naar onze PROCESSING-resolutie
    (preview = rgb_width x rgb_height), want de fabrieksmatrix hoort bij de
    volledige sensor-resolutie. getCameraIntrinsics(socket, w, h) doet die
    schaling intern voor je.

    De focal length fx/fy (in pixels) is de spil van alle pixel->mm
    berekeningen verderop.
    """
    calib = device.readCalibration()
    # Intrinsics voor CAM_A (RGB), geschaald naar onze preview-resolutie.
    matrix = calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, cfg.rgb_width, cfg.rgb_height
    )
    fx = matrix[0][0]
    fy = matrix[1][1]
    cx = matrix[0][2]
    cy = matrix[1][2]
    print(
        f"[CALIB] OAK-D intrinsics @ {cfg.rgb_width}x{cfg.rgb_height}: "
        f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
    )
    return Intrinsics(fx, fy, cx, cy, cfg.rgb_width, cfg.rgb_height)


# ======================================================================
# WISKUNDE  -  pixel <-> millimeter via pinhole-model
# ======================================================================
def pixels_to_mm(pixel_size: float, z_meters: float, focal_px: float) -> float:
    """
    Zet een afstand in beeld-pixels om naar millimeters in de echte wereld.

    Pinhole-model: een object met reele grootte W op afstand Z projecteert
    op p pixels volgens:
            p = focal_px * W / Z
    Daaruit volgt de inverse (van pixels terug naar meters):
            W = p * Z / focal_px
    Met Z in meters en focal_px in pixels is W in meters; *1000 -> mm.

    Doordat Z (de live catenary-afstand) in de formule zit, is de meting
    AFSTAND-ONAFHANKELIJK: hangt de kabel dichterbij (kleinere Z) dan worden
    de pixels automatisch met een kleinere factor naar mm geschaald.
    """
    if focal_px <= 0:
        return 0.0
    meters = (pixel_size * z_meters) / focal_px
    return meters * 1000.0


# ======================================================================
# METING-HELPERS
# ======================================================================
def median_depth_at(depth_frame: np.ndarray, x0: int, x1: int,
                    y0: int, y1: int) -> float:
    """
    Robuuste mediaan-Z (in meters) over een ROI van de depth-map.

    Depth komt uit StereoDepth in millimeters (uint16); 0 = ongeldig.
    We negeren nullen en nemen de mediaan -> ongevoelig voor ruis/gaten.
    """
    h, w = depth_frame.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = depth_frame[y0:y1, x0:x1]
    valid = roi[roi > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) / 1000.0  # mm -> m


def measure_cable_thickness_px(gray: np.ndarray, cfg: Config):
    """
    Meet de verticale dikte van de kabel in pixels via randdetectie.

    Aanname: de kabel loopt ~horizontaal door beeld. Per beeldkolom in een
    centrale strook zoeken we de bovenste en onderste rand (Canny), en nemen
    de mediaan van (onder - boven) als robuuste pixel-dikte. Tevens geven we
    het verticale midden (y-center) terug voor de catenary/Y-check.

    Returns: (thickness_px, y_top, y_bottom, y_center_frac) of None.
    """
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)

    # Bekijk alleen een centrale verticale strook kolommen -> stabieler en
    # ongevoelig voor randen aan de beeldranden.
    col_start, col_end = int(w * 0.30), int(w * 0.70)

    thicknesses = []
    tops, bottoms = [], []
    for x in range(col_start, col_end, 4):  # om de 4 kolommen = sneller
        ys = np.where(edges[:, x] > 0)[0]
        if ys.size >= 2:
            y_top, y_bot = ys[0], ys[-1]
            span = y_bot - y_top
            # Filter onzin: te dun (ruis) of bijna volledige beeldhoogte.
            if 5 < span < h * 0.9:
                thicknesses.append(span)
                tops.append(y_top)
                bottoms.append(y_bot)

    if len(thicknesses) < 5:
        return None

    thickness_px = float(np.median(thicknesses))
    y_top = float(np.median(tops))
    y_bottom = float(np.median(bottoms))
    y_center_frac = ((y_top + y_bottom) / 2.0) / h
    return thickness_px, y_top, y_bottom, y_center_frac


# ======================================================================
# 1. ODOMETER  -  snelheid uit sparse optical flow + Z
# ======================================================================
class Odometer:
    """
    Berekent lijnsnelheid (cm/s) en totale afgelegde afstand (m).

    De FeatureTracker geeft per frame sparse features met een stabiel ID.
    Door een feature-ID tussen twee frames te volgen kennen we de
    pixelverplaatsing dp. Die zetten we via het pinhole-model + live Z om
    naar een reele verplaatsing:
            dX = dp * Z / fx
    Snelheid = dX / dt (dt uit de device-timestamps). De kabel beweegt
    overwegend langs 1 as; we nemen de mediaan van de per-feature snelheden
    -> robuust tegen uitschieters en achtergrond.
    """

    def __init__(self, intr: Intrinsics, cfg: Config):
        self.intr = intr
        self.cfg = cfg
        self.prev_features = {}          # id -> (x, y)
        self.prev_ts = None
        self.total_distance_m = 0.0
        self.speed_hist = deque(maxlen=cfg.smooth_window)

    @staticmethod
    def _to_dict(tracked_features):
        out = {}
        for f in tracked_features:
            out[f.id] = (f.position.x, f.position.y)
        return out

    def update(self, tracked_features, timestamp: float, z_meters: float):
        cur = self._to_dict(tracked_features)

        if self.prev_ts is None or z_meters <= 0:
            self.prev_features = cur
            self.prev_ts = timestamp
            return self.smoothed_speed()

        dt = timestamp - self.prev_ts
        if dt <= 1e-4:
            return self.smoothed_speed()

        # Per gematchte feature de reele verplaatsing bepalen.
        displacements_m = []
        for fid, (x, y) in cur.items():
            if fid in self.prev_features:
                px, py = self.prev_features[fid]
                dp = math.hypot(x - px, y - py)  # pixelverplaatsing
                # Pinhole: meters = pixels * Z / fx  (afstand-gecompenseerd)
                dX = (dp * z_meters) / self.intr.fx
                displacements_m.append(dX)

        if displacements_m:
            # Mediaan-verplaatsing -> robuuste, dominante beweging.
            median_dX = float(np.median(displacements_m))
            speed_cms = (median_dX / dt) * 100.0      # m/s -> cm/s
            self.speed_hist.append(speed_cms)
            # Afstand integreren over de gemeten verplaatsing.
            self.total_distance_m += median_dX

        self.prev_features = cur
        self.prev_ts = timestamp
        return self.smoothed_speed()

    def smoothed_speed(self) -> float:
        if not self.speed_hist:
            return 0.0
        return float(np.median(self.speed_hist))


# ======================================================================
# 3. ALARM-LOGICA
# ======================================================================
# Status-codes
STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_CRIT = "CRIT"

STATUS_COLOR = {
    STATUS_OK:   (0, 200, 0),      # GROEN
    STATUS_WARN: (0, 165, 255),    # ORANJE
    STATUS_CRIT: (0, 0, 255),      # ROOD
}
STATUS_LABEL = {
    STATUS_OK:   "OK",
    STATUS_WARN: "WAARSCHUWING - tape/afwijking",
    STATUS_CRIT: "KRITIEK - STOP",
}


def evaluate_alarms(diameter_mm: float, baseline_mm: float,
                    z_m: float, y_center_frac: float, cfg: Config):
    """
    Bepaalt de status + reden op basis van dikte-afwijking en catenary.

    - Dikte-afwijking t.o.v. baseline:
        >crit_deviation (15%) -> ROOD  (birdcage / dikke tape-bult)
        >warn_deviation (10%) -> ORANJE
    - Catenary: Z buiten [z_min, z_max] of Y-center buiten band
        -> ROOD (kabel te strak/slap of uit de geleiderrol).

    Returns: (status, reden_string)
    """
    reasons = []
    status = STATUS_OK

    # --- Catenary / doorhang (Z) ---
    if z_m > 0:
        if z_m < cfg.z_min_m:
            status = STATUS_CRIT
            reasons.append(f"Kabel te SLAP/dichtbij (Z={z_m:.2f}m)")
        elif z_m > cfg.z_max_m:
            status = STATUS_CRIT
            reasons.append(f"Kabel te STRAK/ver (Z={z_m:.2f}m)")

    # --- Verticale positie (Y) -> uit geleiderrol gelopen ---
    if not (cfg.y_center_min <= y_center_frac <= cfg.y_center_max):
        status = STATUS_CRIT
        reasons.append(f"Kabel-as buiten band (Y={y_center_frac:.2f})")

    # --- Dikte-afwijking ---
    if baseline_mm > 0 and diameter_mm > 0:
        deviation = abs(diameter_mm - baseline_mm) / baseline_mm
        if deviation > cfg.crit_deviation:
            status = STATUS_CRIT
            reasons.append(
                f"Dikte +-{deviation*100:.0f}% (birdcage?) "
                f"{diameter_mm:.0f}mm vs {baseline_mm:.0f}mm"
            )
        elif deviation > cfg.warn_deviation and status != STATUS_CRIT:
            status = STATUS_WARN
            reasons.append(f"Dikte-afwijking {deviation*100:.0f}% (tape?)")

    return status, "; ".join(reasons) if reasons else "Alles binnen tolerantie"


# ======================================================================
# OVERLAY / DISPLAY
# ======================================================================
def draw_overlay(frame, distance_m, speed_cms, diameter_mm, baseline_mm,
                 z_m, status, reason, y_top=None, y_bottom=None):
    h, w = frame.shape[:2]
    color = STATUS_COLOR[status]

    # Halftransparante header-balk
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Meterstand : {distance_m:7.2f} m", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Snelheid   : {speed_cms:6.1f} cm/s", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Dikte      : {diameter_mm:6.1f} mm  (basis {baseline_mm:.0f})",
                (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(frame, f"Z={z_m:.2f}m", (w - 130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Status-indicator (gekleurde stip + label) onderaan
    cv2.rectangle(frame, (0, h - 34), (w, h), (30, 30, 30), -1)
    cv2.circle(frame, (20, h - 17), 10, color, -1)
    cv2.putText(frame, f"{STATUS_LABEL[status]}: {reason}", (40, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Visualiseer de gemeten kabel-randen
    if y_top is not None and y_bottom is not None:
        yt, yb = int(y_top), int(y_bottom)
        cv2.line(frame, (int(w * 0.30), yt), (int(w * 0.70), yt), color, 1)
        cv2.line(frame, (int(w * 0.30), yb), (int(w * 0.70), yb), color, 1)

    # Kleur-rand rond het beeld als snelle visuele status
    cv2.rectangle(frame, (1, 1), (w - 2, h - 2), color, 3)
    return frame


# ======================================================================
# HOOFD-LOOP
# ======================================================================
def main(cfg: Config = CFG):
    pipeline = build_pipeline(cfg)

    with dai.Device(pipeline) as device:
        # --- Kalibratie live uit de chip ---
        intr = read_intrinsics(device, cfg)

        # --- Output-queues (non-blocking, kleine buffers tegen latency) ---
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        q_feat = device.getOutputQueue("features", maxSize=4, blocking=False)

        odometer = Odometer(intr, cfg)

        # Baseline-diameter: of vast (config) of auto-gekalibreerd.
        baseline_mm = cfg.nominal_diameter_mm
        baseline_samples = []
        calibrating = cfg.autocalibrate_baseline

        last_mqtt = 0.0
        latest_depth = None
        latest_features = []

        print("[INFO] PoC draait. Druk op 'q' in het venster om te stoppen.")

        while True:
            in_rgb = q_rgb.tryGet()
            in_depth = q_depth.tryGet()
            in_feat = q_feat.tryGet()

            if in_depth is not None:
                latest_depth = in_depth.getFrame()           # uint16, mm
            if in_feat is not None:
                latest_features = in_feat.trackedFeatures

            if in_rgb is None:
                # Niets nieuws van de RGB-stream; klein beetje yield.
                if cv2.waitKey(1) == ord("q"):
                    break
                continue

            frame = in_rgb.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]

            # device-timestamp -> stabiele dt voor snelheidsberekening
            ts = in_rgb.getTimestamp().total_seconds()

            # ---- Centrale Z (catenary-afstand) bepalen ----
            # Mediaan-depth in een centrale ROI = afstand camera -> kabel.
            z_m = 0.0
            if latest_depth is not None:
                z_m = median_depth_at(
                    latest_depth,
                    int(w * 0.35), int(w * 0.65),
                    int(h * 0.35), int(h * 0.65),
                )

            # ---- 1. Odometer & snelheid ----
            speed_cms = odometer.update(latest_features, ts, z_m)
            distance_m = odometer.total_distance_m

            # ---- 2. Diktemeting (catenary-gecorrigeerd) ----
            diameter_mm = 0.0
            y_top = y_bottom = None
            y_center_frac = 0.5
            meas = measure_cable_thickness_px(gray, cfg)
            if meas is not None and z_m > 0:
                thickness_px, y_top, y_bottom, y_center_frac = meas
                # Pixel-dikte -> mm via fy (verticale focal length) + live Z.
                diameter_mm = pixels_to_mm(thickness_px, z_m, intr.fy)

                # Auto-kalibratie van de basisdikte op de eerste goede frames.
                if calibrating and diameter_mm > 0:
                    baseline_samples.append(diameter_mm)
                    if len(baseline_samples) >= cfg.autocalibrate_frames:
                        baseline_mm = float(np.median(baseline_samples))
                        calibrating = False
                        print(f"[CALIB] Baseline-diameter = {baseline_mm:.1f} mm")

            # ---- 3. Alarm-logica ----
            status, reason = evaluate_alarms(
                diameter_mm, baseline_mm, z_m, y_center_frac, cfg
            )
            if calibrating:
                status, reason = STATUS_OK, "Baseline kalibreren..."

            # ROOD -> noodstop-koppeling aanroepen.
            if status == STATUS_CRIT:
                trigger_emergency_stop(reason)

            # ---- MQTT-telemetrie (max ~2 Hz) ----
            now = time.time()
            if now - last_mqtt > 0.5:
                send_mqtt_update({
                    "distance_m": distance_m,
                    "speed_cms": speed_cms,
                    "diameter_mm": diameter_mm,
                    "baseline_mm": baseline_mm,
                    "z_m": z_m,
                    "status": status,
                    "reason": reason,
                })
                last_mqtt = now

            # ---- Display ----
            frame = draw_overlay(
                frame, distance_m, speed_cms, diameter_mm, baseline_mm,
                z_m, status, reason, y_top, y_bottom,
            )
            cv2.imshow("Kabel-inspectie PoC (q=stop)", frame)

            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
    print("[INFO] Gestopt.")


if __name__ == "__main__":
    main()
