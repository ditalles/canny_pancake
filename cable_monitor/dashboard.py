"""
Flask operator-dashboard.

Toont live de catenary-monitor: status-banner, meet-tegels, zicht-/health,
een geannoteerde videostream (MJPEG) en een grafiek van dikte/zicht. Bedoeld
zodat een operator kan MEEKIJKEN en zelf beslist in te grijpen (geen
machinebesturing).

Routes:
  /              dashboard-pagina
  /api/state     JSON-snapshot (polling-fallback)
  /stream        Server-Sent Events met live JSON
  /video         MJPEG van het geannoteerde beeld
"""

from __future__ import annotations

import json
import threading
import time

from flask import Flask, Response, jsonify, render_template

from .config import Config
from .state import MonitorState


def create_app(state: MonitorState, cfg: Config) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("dashboard.html", cfg=cfg)

    @app.route("/api/state")
    def api_state():
        return jsonify(state.snapshot())

    @app.route("/stream")
    def stream():
        def gen():
            while True:
                data = json.dumps(state.snapshot())
                yield f"data: {data}\n\n"
                time.sleep(0.4)   # ~2.5 Hz update naar de browser
        return Response(gen(), mimetype="text/event-stream")

    @app.route("/video")
    def video():
        def gen():
            boundary = b"--frame"
            while True:
                jpeg = state.get_jpeg()
                if jpeg is not None:
                    yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpeg + b"\r\n")
                time.sleep(0.066)  # ~15 fps naar de browser
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def run_dashboard_in_thread(state: MonitorState, cfg: Config) -> threading.Thread:
    """Start de Flask-server in een daemon-thread naast de engine."""
    app = create_app(state, cfg)

    def _serve():
        # threaded=True zodat /video, /stream en /api parallel bediend worden.
        app.run(host=cfg.dashboard_host, port=cfg.dashboard_port,
                threaded=True, debug=False, use_reloader=False)

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t
