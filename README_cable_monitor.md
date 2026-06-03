# Catenary kabel-monitor (PoC)

Robuuste, **monitoring-only** catenary-monitor met anomaliedetectie voor een
zware maritieme kabel die via een geleiderrol de kade op loopt. Draait op een
OAK-D Lite (DepthAI) en toont alles live op een Flask-operator-dashboard.

> **Geen machinebesturing.** Dit systeem meet, beoordeelt en visualiseert.
> De operator beslist of ingrijpen nodig is. Er zit bewust geen noodstop/
> actuator in (geen SIL/PL veiligheidsfunctie).

## Wat het doet
1. **Odometer & snelheid** — sparse optical flow (FeatureTracker) + live Z
   (StereoDepth) → snelheid (cm/s) en meterstand (m).
2. **Catenary-gecorrigeerde diktemeting** — verticale randdetectie omgerekend
   naar mm via de focal length én de live Z (afstand-onafhankelijk).
3. **Anomalie- & catenary-alarm** — dikte-afwijking (>10% WARN, >15% CRIT,
   bv. tape/birdcage) en doorhang buiten Z/Y-venster.
4. **Health-aware robuustheid** — bij regen/sneeuw/mist/nacht/beslagen lens
   daalt de zicht-score en worden anomalie-alarmen onderdrukt en als
   **ZICHT ONBETROUWBAAR** getoond (eerlijk i.p.v. vals).

## Snel starten
```bash
pip install -r requirements.txt

# Demo zonder camera (simulator + dashboard):
python run_monitor.py --simulate
#   -> open http://localhost:5006

# Echte OAK-D Lite:
python run_monitor.py --source depthai
```

Opties: `--no-dashboard` (headless), `--window` (lokaal OpenCV-venster),
`--port N` (dashboard-poort).

## Architectuur (modulair, source-agnostic)
```
run_monitor.py            entrypoint / CLI
cable_monitor/
  config.py               alle drempels & parameters
  sources.py              FrameSource: DepthAISource | SimulatedSource  (+ Intrinsics)
  health.py               zicht-/betrouwbaarheidsscore (GOOD/DEGRADED/BLIND)
  perception.py           odometer, diktemeting, pixel->mm, anomalie-beoordeling
  state.py                thread-safe gedeelde toestand + historie
  engine.py               koppelt bron -> perceptie -> health -> toestand
  dashboard.py            Flask: /, /api/state, /stream (SSE), /video (MJPEG)
  templates/dashboard.html operator-UI (tegels, banner, video, grafieken)
```
Een toekomstige **IR/nacht-camera** is simpelweg een nieuwe `FrameSource`;
de rest blijft ongewijzigd.

## Camera-kalibratie (focal length uit de chip)
`DepthAISource` leest de fabriekskalibratie live uit de EEPROM:
`device.readCalibration().getCameraIntrinsics(CAM_A, w, h)` levert fx/fy/cx/cy
geschaald naar de processing-resolutie. fx/fy (px) is de spil van de
pixel→mm-omrekening.

## Documentatie
- `docs/montage_checklist.md` — installatie & omgeving (weer/nacht/maritiem).
- `docs/positie_eisen.md` — eisen aan het meetpunt + wat nodig is om de plek
  samen te kiezen.
