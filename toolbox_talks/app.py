"""Toolbox-talk attendance & sign-off app.

A toolbox talk is a short safety briefing (common on construction sites and in
factories). The law / safety auditors require a record proving each worker
attended *and understood* the topic — traditionally a paper sheet everyone
signs.

This app replaces that sheet:

  1. A supervisor creates a talk (topic, presenter, key points).
  2. The app makes a QR code unique to that talk.
  3. Workers scan it, read the topic + key points, enter their name and tick
     "I attended and understood" — that confirmation is their signature.
  4. The supervisor gets a timestamped, exportable attendance/sign-off record.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5007/
"""

import csv
import io
import os
import secrets
import sqlite3
from datetime import datetime, date

import qrcode
import qrcode.image.svg
from flask import (
    Flask,
    Response,
    abort,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "toolbox.db")


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
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS talks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            topic      TEXT    NOT NULL,
            presenter  TEXT,
            talk_date  TEXT    NOT NULL,
            key_points TEXT,
            token      TEXT    NOT NULL UNIQUE,
            created_at TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS signoffs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            talk_id    INTEGER NOT NULL REFERENCES talks(id),
            name       TEXT    NOT NULL,
            understood INTEGER NOT NULL DEFAULT 0,
            signed_at  TEXT    NOT NULL,
            ip         TEXT
        );
        """
    )
    db.commit()
    db.close()


# --- Admin: list & create talks --------------------------------------------


@app.route("/")
def index():
    db = get_db()
    talks = db.execute(
        """
        SELECT t.*, COUNT(s.id) AS attendees
        FROM talks t LEFT JOIN signoffs s ON s.talk_id = t.id
        GROUP BY t.id ORDER BY t.id DESC
        """
    ).fetchall()
    return render_template("index.html", talks=talks)


@app.route("/talk/new", methods=["GET", "POST"])
def new_talk():
    if request.method == "POST":
        topic = (request.form.get("topic") or "").strip()
        if not topic:
            return render_template("new_talk.html", error="A topic is required.",
                                   today=date.today().isoformat())
        db = get_db()
        cur = db.execute(
            "INSERT INTO talks (topic, presenter, talk_date, key_points, token, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                topic,
                (request.form.get("presenter") or "").strip(),
                request.form.get("talk_date") or date.today().isoformat(),
                (request.form.get("key_points") or "").strip(),
                secrets.token_urlsafe(9),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()
        return redirect(url_for("talk_detail", talk_id=cur.lastrowid))
    return render_template("new_talk.html", today=date.today().isoformat())


@app.route("/talk/<int:talk_id>")
def talk_detail(talk_id):
    db = get_db()
    talk = db.execute("SELECT * FROM talks WHERE id = ?", (talk_id,)).fetchone()
    if talk is None:
        abort(404)
    signoffs = db.execute(
        "SELECT name, understood, signed_at FROM signoffs "
        "WHERE talk_id = ? ORDER BY id DESC",
        (talk_id,),
    ).fetchall()
    return render_template("talk_detail.html", talk=talk, signoffs=signoffs)


@app.route("/talk/<int:talk_id>/qr.svg")
def talk_qr(talk_id):
    db = get_db()
    talk = db.execute("SELECT token FROM talks WHERE id = ?", (talk_id,)).fetchone()
    if talk is None:
        abort(404)
    attend_url = url_for("attend", token=talk["token"], _external=True)
    img = qrcode.make(attend_url, image_factory=qrcode.image.svg.SvgImage)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), mimetype="image/svg+xml")


@app.route("/talk/<int:talk_id>/export.csv")
def export_csv(talk_id):
    db = get_db()
    talk = db.execute("SELECT * FROM talks WHERE id = ?", (talk_id,)).fetchone()
    if talk is None:
        abort(404)
    rows = db.execute(
        "SELECT name, understood, signed_at, ip FROM signoffs "
        "WHERE talk_id = ? ORDER BY id",
        (talk_id,),
    ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([f"Toolbox talk: {talk['topic']}"])
    writer.writerow([f"Presenter: {talk['presenter']}  Date: {talk['talk_date']}"])
    writer.writerow([])
    writer.writerow(["Name", "Attended & understood", "Signed at", "IP address"])
    for r in rows:
        writer.writerow(
            [r["name"], "Yes" if r["understood"] else "No", r["signed_at"], r["ip"]]
        )

    safe_topic = "".join(c if c.isalnum() else "_" for c in talk["topic"])[:40]
    filename = f"toolbox_{safe_topic}_{talk['talk_date']}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# --- Worker: scan & sign off ------------------------------------------------


@app.route("/attend/<token>", methods=["GET", "POST"])
def attend(token):
    db = get_db()
    talk = db.execute("SELECT * FROM talks WHERE token = ?", (token,)).fetchone()
    if talk is None:
        abort(404)

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        understood = 1 if request.form.get("understood") else 0
        if not name:
            return render_template("attend.html", talk=talk,
                                   error="Please enter your name.")
        if not understood:
            return render_template(
                "attend.html", talk=talk,
                error="Please tick the box to confirm you attended and understood.",
                name=name,
            )

        existing = db.execute(
            "SELECT signed_at FROM signoffs WHERE talk_id = ? AND name = ?",
            (talk["id"], name),
        ).fetchone()
        if existing:
            return render_template("attend.html", talk=talk, already=True,
                                   name=name, time=existing["signed_at"])

        signed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.execute(
            "INSERT INTO signoffs (talk_id, name, understood, signed_at, ip) "
            "VALUES (?, ?, ?, ?, ?)",
            (talk["id"], name, understood, signed_at, request.remote_addr),
        )
        db.commit()
        return render_template("attend.html", talk=talk, success=True,
                               name=name, time=signed_at)

    return render_template("attend.html", talk=talk)


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5007)
