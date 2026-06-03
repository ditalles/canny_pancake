"""Centrale configuratie voor de catenary-monitor."""

from dataclasses import dataclass


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Camera / pijplijn
    # ------------------------------------------------------------------
    proc_width: int = 640            # processing/preview breedte (px)
    proc_height: int = 400           # processing/preview hoogte (px)
    fps: int = 30

    # ------------------------------------------------------------------
    # Kabel / fysiek
    # ------------------------------------------------------------------
    nominal_diameter_mm: float = 80.0   # verwachte basis-diameter
    nominal_speed_cms: float = 16.6     # verwachte lijnsnelheid (~10 m/min)

    # ------------------------------------------------------------------
    # Catenary (doorhang) veiligheidsvenster
    #   Te dichtbij  = te slap (doorhang).
    #   Te ver weg   = te strak getrokken.
    # ------------------------------------------------------------------
    z_min_m: float = 0.40
    z_max_m: float = 1.20

    # Verticale positie van de kabel-as in beeld (fractie 0..1). Buiten deze
    # band hangt de kabel scheef / uit de geleiderrol.
    y_center_min: float = 0.25
    y_center_max: float = 0.75

    # ------------------------------------------------------------------
    # Anomalie-drempels (dikte t.o.v. baseline)
    # ------------------------------------------------------------------
    warn_deviation: float = 0.10     # >10% -> WAARSCHUWING (tape-reparatie)
    crit_deviation: float = 0.15     # >15% -> KRITIEK (birdcage / bult)

    # ------------------------------------------------------------------
    # Randdetectie (diktemeting)
    # ------------------------------------------------------------------
    canny_low: int = 60
    canny_high: int = 180
    roi_col_lo: float = 0.30         # centrale kolom-strook voor de meting
    roi_col_hi: float = 0.70

    # ------------------------------------------------------------------
    # Meting-filtering (ruisonderdrukking)
    # ------------------------------------------------------------------
    smooth_window: int = 8

    # ------------------------------------------------------------------
    # Baseline auto-kalibratie
    # ------------------------------------------------------------------
    autocalibrate_baseline: bool = True
    autocalibrate_frames: int = 60

    # ------------------------------------------------------------------
    # HEALTH / zicht-betrouwbaarheid  (robuustheid bij weer/nacht)
    #   Drempels op genormaliseerde 0..1 metrics. Bij overschrijding daalt
    #   de zicht-score en worden anomalie-alarmen onderdrukt (eerlijk i.p.v.
    #   vals).
    # ------------------------------------------------------------------
    min_brightness: float = 0.12     # < -> te donker (nacht/onderbelicht)
    max_brightness: float = 0.97     # > -> overbelicht / verblinding
    min_focus: float = 0.10          # < -> wazig (mist/regen/beslagen lens)
    max_precip: float = 0.45         # > -> te veel regen/sneeuw-speckle
    min_depth_valid: float = 0.20    # < -> te weinig geldige depth-pixels
    min_features: int = 12           # < -> optical flow onbetrouwbaar
    degraded_visibility: float = 0.65  # < -> DEGRADED
    blind_visibility: float = 0.35     # < -> BLIND (zicht onbetrouwbaar)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 5006
    history_len: int = 240           # samples in de dashboard-grafiek

    # ------------------------------------------------------------------
    # Telemetrie (advies, geen besturing)
    # ------------------------------------------------------------------
    mqtt_enabled: bool = False
    mqtt_interval_s: float = 0.5
