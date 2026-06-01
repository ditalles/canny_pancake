"""QR-code attendance check-in app.

A single QR code is printed/displayed at the door. People scan it, the
check-in page opens, they confirm their name and tap "Check in". The server
records who checked in and the exact time in a SQLite database. Staff can view
a live list and export it to CSV.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5006/admin to see the QR code and live list.
"""

import csv
import io
import os
import sqlite3
from datetime import datetime, date

import qrcode
import qrcode.image.svg
from flask import (
    Flask,
    Response,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")

# A secret-ish code embedded in the QR link so random people who guess the URL
# can't check themselves in. Change this and reprint the QR to "rotate" it.
# In production, read it from an environment variable instead of hard-coding.
EVENT_CODE = os.environ.get("ATTENDANCE_EVENT_CODE", "doorcode2026")

# Optional: a fixed roster shown as quick-pick buttons. Leave empty to let
# people type their own name freely.
ROSTER = [
    name.strip()
    for name in os.environ.get("ATTENDANCE_ROSTER", "").split(",")
    if name.strip()
]


# --- Database helpers -------------------------------------------------------


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL,
            checkin_time TEXT    NOT NULL,
            checkin_date TEXT    NOT NULL,
            ip           TEXT
        )
        """
    )
    db.commit()
    db.close()


# --- Routes -----------------------------------------------------------------


@app.route("/")
def index():
    return redirect(url_for("checkin"))


@app.route("/checkin", methods=["GET", "POST"])
def checkin():
    # Reject anyone who didn't come through the real QR link.
    if request.args.get("code") != EVENT_CODE and request.form.get("code") != EVENT_CODE:
        return render_template("checkin.html", invalid=True, roster=ROSTER,
                               code=EVENT_CODE)

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        if not name:
            return render_template("checkin.html", error="Please enter your name.",
                                   roster=ROSTER, code=EVENT_CODE)

        today = date.today().isoformat()
        db = get_db()

        # Friendly note if they already checked in today (still recorded once).
        existing = db.execute(
            "SELECT checkin_time FROM attendance "
            "WHERE name = ? AND checkin_date = ?",
            (name, today),
        ).fetchone()
        if existing:
            return render_template(
                "checkin.html",
                already=True,
                name=name,
                time=existing["checkin_time"],
                roster=ROSTER,
                code=EVENT_CODE,
            )

        now = datetime.now()
        db.execute(
            "INSERT INTO attendance (name, checkin_time, checkin_date, ip) "
            "VALUES (?, ?, ?, ?)",
            (name, now.strftime("%H:%M:%S"), today, request.remote_addr),
        )
        db.commit()
        return render_template(
            "checkin.html",
            success=True,
            name=name,
            time=now.strftime("%H:%M:%S"),
            roster=ROSTER,
            code=EVENT_CODE,
        )

    return render_template("checkin.html", roster=ROSTER, code=EVENT_CODE)


@app.route("/admin")
def admin():
    today = date.today().isoformat()
    db = get_db()
    records = db.execute(
        "SELECT name, checkin_time FROM attendance "
        "WHERE checkin_date = ? ORDER BY id DESC",
        (today,),
    ).fetchall()
    return render_template("admin.html", records=records, today=today)


@app.route("/qr.svg")
def qr_svg():
    """Return the door QR code as a scalable SVG (prints crisply at any size)."""
    checkin_url = url_for("checkin", code=EVENT_CODE, _external=True)
    img = qrcode.make(checkin_url, image_factory=qrcode.image.svg.SvgImage)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), mimetype="image/svg+xml")


@app.route("/export.csv")
def export_csv():
    """Download all attendance records as a CSV (opens in Excel)."""
    db = get_db()
    rows = db.execute(
        "SELECT checkin_date, checkin_time, name, ip FROM attendance "
        "ORDER BY checkin_date DESC, id DESC"
    ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Time", "Name", "IP address"])
    for r in rows:
        writer.writerow([r["checkin_date"], r["checkin_time"], r["name"], r["ip"]])

    filename = f"attendance_{date.today().isoformat()}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    init_db()
    # host=0.0.0.0 so phones on the same Wi-Fi can reach it.
    app.run(debug=True, host="0.0.0.0", port=5006)
