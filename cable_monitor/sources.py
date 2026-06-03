"""
Beeldbronnen (source-agnostic laag).

Elke bron levert een uniforme `FrameBundle` (RGB + depth + sparse features +
intrinsics + timestamp). Daardoor draait dezelfde perceptie-code op:
  - `DepthAISource`     -> echte OAK-D Lite
  - `SimulatedSource`   -> synthetische data om het dashboard te demonstreren
                           zonder camera (incl. anomalieen en slecht-zicht)
Een toekomstige IR/nacht-bron is simpelweg een nieuwe `FrameSource`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config


# ----------------------------------------------------------------------
@dataclass
class Intrinsics:
    """Camera-intrinsics, geschaald naar de processing-resolutie."""
    fx: float           # focal length X (pixels)
    fy: float           # focal length Y (pixels)
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class FrameBundle:
    """Alles wat de perceptie per frame nodig heeft."""
    rgb: np.ndarray                         # BGR uint8 (H, W, 3)
    depth_mm: Optional[np.ndarray]          # uint16 mm, 0 = ongeldig
    features: List[Tuple[int, float, float]]  # (id, x, y) sparse optical flow
    timestamp: float                        # seconden (monotone bron-tijd)
    intrinsics: Intrinsics


class FrameSource:
    """Interface voor een beeldbron."""

    def open(self) -> "FrameSource":
        return self

    def read(self) -> Optional[FrameBundle]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


# ======================================================================
# DepthAI / OAK-D Lite
# ======================================================================
class DepthAISource(FrameSource):
    """
    EEN DepthAI-pijplijn: RGB (CAM_A) + StereoDepth (uitgelijnd op CAM_A) +
    FeatureTracker op de RGB-feed. Alleen RGB + depth + sparse features gaan
    over USB -> lichte belasting voor een laptop/industrie-PC.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._device = None
        self._q_rgb = None
        self._q_depth = None
        self._q_feat = None
        self._intr: Optional[Intrinsics] = None
        self._latest_depth = None
        self._latest_features: List[Tuple[int, float, float]] = []

    def open(self) -> "DepthAISource":
        import depthai as dai  # lokaal: alleen nodig voor echte hardware

        cfg = self.cfg
        pipeline = dai.Pipeline()

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setPreviewSize(cfg.proc_width, cfg.proc_height)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(cfg.fps)

        mono_l = pipeline.create(dai.node.MonoCamera)
        mono_r = pipeline.create(dai.node.MonoCamera)
        for mono, sock in (
            (mono_l, dai.CameraBoardSocket.CAM_B),
            (mono_r, dai.CameraBoardSocket.CAM_C),
        ):
            mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
            mono.setBoardSocket(sock)
            mono.setFps(cfg.fps)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # depth in RGB-frame
        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)

        feat = pipeline.create(dai.node.FeatureTracker)
        feat.initialConfig.setNumTargetFeatures(256)
        feat.initialConfig.setMotionEstimator(True)
        cam.video.link(feat.inputImage)

        xrgb = pipeline.create(dai.node.XLinkOut); xrgb.setStreamName("rgb")
        xdep = pipeline.create(dai.node.XLinkOut); xdep.setStreamName("depth")
        xfea = pipeline.create(dai.node.XLinkOut); xfea.setStreamName("features")
        cam.preview.link(xrgb.input)
        stereo.depth.link(xdep.input)
        feat.outputFeatures.link(xfea.input)

        self._device = dai.Device(pipeline)
        self._q_rgb = self._device.getOutputQueue("rgb", 4, False)
        self._q_depth = self._device.getOutputQueue("depth", 4, False)
        self._q_feat = self._device.getOutputQueue("features", 4, False)

        # --- Intrinsics live uit de EEPROM van de chip ---
        # getCameraIntrinsics(socket, w, h) schaalt de fabrieksmatrix naar
        # onze processing-resolutie. fx/fy (px) is de spil van pixel->mm.
        calib = self._device.readCalibration()
        m = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, cfg.proc_width, cfg.proc_height
        )
        self._intr = Intrinsics(
            fx=m[0][0], fy=m[1][1], cx=m[0][2], cy=m[1][2],
            width=cfg.proc_width, height=cfg.proc_height,
        )
        return self

    def read(self) -> Optional[FrameBundle]:
        in_depth = self._q_depth.tryGet()
        in_feat = self._q_feat.tryGet()
        if in_depth is not None:
            self._latest_depth = in_depth.getFrame()
        if in_feat is not None:
            self._latest_features = [
                (f.id, f.position.x, f.position.y) for f in in_feat.trackedFeatures
            ]

        in_rgb = self._q_rgb.tryGet()
        if in_rgb is None:
            return None

        return FrameBundle(
            rgb=in_rgb.getCvFrame(),
            depth_mm=self._latest_depth,
            features=list(self._latest_features),
            timestamp=in_rgb.getTimestamp().total_seconds(),
            intrinsics=self._intr,
        )

    def close(self) -> None:
        if self._device is not None:
            self._device.close()


# ======================================================================
# Simulator  -  draait het hele dashboard zonder camera
# ======================================================================
class SimulatedSource(FrameSource):
    """
    Genereert synthetische frames met een horizontale 'kabel', bewegende
    achtergrond-features (voor de odometer), een depth-map, en periodieke
    gebeurtenissen:
      - birdcage  (dikte +25%)         -> verwacht KRITIEK
      - tape      (dikte +12%)         -> verwacht WAARSCHUWING
      - te strak  (Z buiten venster)   -> verwacht KRITIEK
      - slecht zicht (donker + speckle)-> verwacht 'zicht onbetrouwbaar'
    """

    def __init__(self, cfg: Config, seed: int = 7):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.t0 = time.time()
        self.frame_idx = 0
        self.W, self.H = cfg.proc_width, cfg.proc_height
        # Synthetische intrinsics ~ OAK-D Lite bij deze resolutie.
        self._intr = Intrinsics(fx=460.0, fy=460.0,
                                cx=self.W / 2, cy=self.H / 2,
                                width=self.W, height=self.H)
        # Achtergrond-features die horizontaal mee scrollen met de kabel.
        self._feat_x = self.rng.uniform(0, self.W, size=90)
        self._feat_y = self.rng.uniform(0, self.H, size=90)
        self._feat_id = np.arange(90)
        # Laag-frequente achtergrond-textuur: genoeg gradient voor 'scherpte'
        # maar geen harde Canny-randen overal (anders verdrinkt de kabelrand).
        coarse = self.rng.integers(30, 80, size=(self.H // 16, self.W // 16),
                                   dtype=np.uint8)
        bg = cv2.resize(coarse, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        self._bg = cv2.GaussianBlur(bg, (0, 0), 3)

    def read(self) -> Optional[FrameBundle]:
        cfg = self.cfg
        t = time.time() - self.t0
        self.frame_idx += 1
        dt = 1.0 / cfg.fps

        # --- Catenary-afstand Z: langzaam varierend rond het midden ---
        z_mid = (cfg.z_min_m + cfg.z_max_m) / 2
        z = z_mid + 0.12 * math.sin(t * 0.25)
        # Periodieke 'te strak' gebeurtenis.
        if 18.0 < (t % 30.0) < 21.0:
            z = cfg.z_max_m + 0.15

        # --- Dikte-gebeurtenissen ---
        diameter_mm = cfg.nominal_diameter_mm
        if 8.0 < (t % 30.0) < 11.0:
            diameter_mm *= 1.25          # birdcage -> KRITIEK
        elif 13.0 < (t % 30.0) < 15.0:
            diameter_mm *= 1.12          # tape -> WAARSCHUWING

        # Pinhole projectie: pixel-dikte = fy * D / Z  (D, Z in meters)
        thickness_px = self._intr.fy * (diameter_mm / 1000.0) / z

        # --- Beweging van de kabel (odometer): px/frame uit echte snelheid ---
        # dp = v * fx / Z / fps   (v in m/s)
        v_ms = cfg.nominal_speed_cms / 100.0
        dp = v_ms * self._intr.fx / z / cfg.fps

        # --- Bouw het beeld ---
        img = np.dstack([self._bg, self._bg, self._bg]).astype(np.uint8)
        # Scroll de achtergrond zodat features echte beweging zien.
        shift = int(self.frame_idx * dp) % self.W
        img = np.roll(img, -shift, axis=1)

        y_center = self.H * 0.5
        half = thickness_px / 2.0
        y0 = int(max(0, y_center - half))
        y1 = int(min(self.H, y_center + half))
        # Lichte kabelband met scherpe DONKERE randlijnen -> sterke, eenduidige
        # Canny-randen die boven de zachte achtergrond uitkomen.
        img[y0:y1, :] = (110, 110, 116)
        img[max(0, y0 - 2):y0 + 1, :] = (10, 10, 10)
        img[y1 - 1:min(self.H, y1 + 2), :] = (10, 10, 10)

        # --- Depth-map (mm) in de kabel-ROI ---
        depth = np.zeros((self.H, self.W), dtype=np.uint16)
        depth[y0:y1, :] = int(z * 1000)
        # Beetje geldige achtergrond-depth eromheen.
        depth[:y0, :] = int((z + 0.3) * 1000)
        depth[y1:, :] = int((z + 0.3) * 1000)

        # --- Bewegende features (id, x, y) ---
        self._feat_x = (self._feat_x - dp) % self.W
        features = [(int(i), float(x), float(y))
                    for i, x, y in zip(self._feat_id, self._feat_x, self._feat_y)]

        # --- Slecht-zicht gebeurtenis: donker + regen/sneeuw-speckle ---
        if 23.0 < (t % 30.0) < 27.0:
            img = (img * 0.18).astype(np.uint8)                 # nacht/onderbel.
            speckle = self.rng.random((self.H, self.W)) > 0.85   # neerslag
            img[speckle] = self.rng.integers(120, 255)
            # Depth wordt onbetrouwbaar in slecht zicht.
            mask = self.rng.random((self.H, self.W)) > 0.4
            depth[mask] = 0
            features = features[:5]                              # weinig flow

        # Realtime tempo nabootsen.
        time.sleep(dt)

        return FrameBundle(
            rgb=img, depth_mm=depth, features=features,
            timestamp=t, intrinsics=self._intr,
        )
