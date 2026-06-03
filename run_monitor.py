#!/usr/bin/env python3
"""
Entrypoint voor de catenary-monitor (monitoring-only).

Voorbeelden
-----------
  # Demo zonder camera: simulator + dashboard (open http://localhost:5006)
  python run_monitor.py --simulate

  # Echte OAK-D Lite + dashboard
  python run_monitor.py --source depthai

  # Met lokaal OpenCV-venster erbij (alleen op een desktop met display)
  python run_monitor.py --simulate --window

  # Zonder dashboard (headless, alleen console/telemetrie)
  python run_monitor.py --source depthai --no-dashboard

Afsluiten: Ctrl+C  (of 'q' in het venster bij --window).
"""

import argparse
import time

from cable_monitor.config import Config
from cable_monitor.dashboard import run_dashboard_in_thread
from cable_monitor.engine import MonitorEngine
from cable_monitor.state import MonitorState


def build_source(kind: str, cfg: Config):
    if kind == "simulate":
        from cable_monitor.sources import SimulatedSource
        return SimulatedSource(cfg)
    if kind == "depthai":
        from cable_monitor.sources import DepthAISource
        return DepthAISource(cfg)
    raise ValueError(f"onbekende bron: {kind}")


def main():
    p = argparse.ArgumentParser(description="Catenary kabel-monitor (monitoring-only)")
    p.add_argument("--source", choices=["depthai", "simulate"], default="depthai",
                   help="beeldbron (default: depthai)")
    p.add_argument("--simulate", action="store_true",
                   help="snelkoppeling voor --source simulate")
    p.add_argument("--no-dashboard", action="store_true",
                   help="geen Flask-dashboard starten")
    p.add_argument("--window", action="store_true",
                   help="lokaal OpenCV-venster tonen (vereist display)")
    p.add_argument("--port", type=int, default=None, help="dashboard-poort")
    args = p.parse_args()

    cfg = Config()
    if args.port:
        cfg.dashboard_port = args.port
    kind = "simulate" if args.simulate else args.source

    state = MonitorState(cfg)
    source = build_source(kind, cfg)
    engine = MonitorEngine(source, state, cfg)

    if not args.no_dashboard:
        run_dashboard_in_thread(state, cfg)
        print(f"[DASHBOARD] http://localhost:{cfg.dashboard_port}  (bron: {kind})")

    try:
        engine.run(show_window=args.window)
    except KeyboardInterrupt:
        print("\n[INFO] Gestopt.")
        engine.stop()


if __name__ == "__main__":
    main()
