"""Toolbox-talk sign-off with passkeys (fingerprint / Face ID / device PIN).

This is the "premium" variant of the toolbox-talk app. Instead of typing a name
and ticking a box, each worker enrols ONCE (types their name and registers a
passkey using their phone's fingerprint / Face ID / PIN). After that, signing
off a talk is: scan the QR -> tap "Sign off" -> fingerprint. No typing.

Why passkeys (WebAuthn)?
  A browser can never read a raw fingerprint. The web standard for this is
  WebAuthn / passkeys: the phone unlocks an on-device private key using the
  biometric/PIN it already trusts, and signs a server challenge. The server
  verifies the signature. That gives cryptographic proof that *this enrolled
  person, on their device, authenticated with their biometric* — stronger than
  a paper signature.

IMPORTANT — secure context:
  WebAuthn only works on https:// or on http://localhost. For real phone use
  over your network you MUST serve this over HTTPS with a real hostname, and
  set RP_ID / ORIGIN below (via env vars) to match that hostname.

Run (local dev):
    pip install -r requirements.txt
    python app.py
    # open http://localhost:5008/
"""

import csv
import functools
import io
import os
import secrets
import sqlite3
from datetime import datetime, date, timedelta

import qrcode
import qrcode.image.svg
from flask import (
    Flask,
    Response,
    abort,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

app = Flask(__name__)
# Session secret: persisted to a file so "remembered" devices survive restarts.
# In production set FLASK_SECRET_KEY in the environment instead.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.permanent_session_lifetime = timedelta(days=365)

DB_PATH = os.path.join(os.path.dirname(__file__), "toolbox_passkey.db")

# Uploaded certificate files are limited in size and to a safe set of types.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
ALLOWED_CERT_MIME = {"image/png", "image/jpeg", "image/webp", "application/pdf"}

# Common qualifications; workers can also pick "Other" and type their own.
CERT_TYPES = [
    "VCA (safety)",
    "Forklift / reach truck",
    "Working at height / climbing",
    "First aid / BHV",
    "Crane / hoisting",
    "Confined space",
    "Welding",
    "Other",
]

# A certificate is flagged when it expires within this many days.
EXPIRY_WARNING_DAYS = 60

# Password for the supervisor (admin) pages. ALWAYS set ADMIN_PASSWORD in the
# environment for real use; the default below is only for local testing.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD") or "admin"
if ADMIN_PASSWORD == "admin":
    print("WARNING: using default admin password 'admin'. "
          "Set ADMIN_PASSWORD before deploying.")


def require_admin(view):
    """Gate a route behind supervisor login (session flag set at /admin/login)."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

# WebAuthn relying-party config. For real (non-localhost) deployment, set these
# to your HTTPS hostname, e.g. RP_ID=talks.example.com ORIGIN=https://talks.example.com
RP_ID = os.environ.get("RP_ID", "localhost")
RP_NAME = os.environ.get("RP_NAME", "Toolbox Talks")
ORIGIN = os.environ.get("ORIGIN", "http://localhost:5008")


# --- Database ---------------------------------------------------------------


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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL, presenter TEXT, talk_date TEXT NOT NULL,
            key_points TEXT, token TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS workers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS credentials (
            cred_id TEXT PRIMARY KEY,
            worker_id INTEGER NOT NULL REFERENCES workers(id),
            public_key TEXT NOT NULL, sign_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS signoffs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            talk_id INTEGER NOT NULL REFERENCES talks(id),
            worker_id INTEGER NOT NULL REFERENCES workers(id),
            name TEXT NOT NULL, signed_at TEXT NOT NULL, ip TEXT
        );
        CREATE TABLE IF NOT EXISTS certificates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id INTEGER NOT NULL REFERENCES workers(id),
            cert_type TEXT NOT NULL, cert_number TEXT,
            issued_date TEXT, expiry_date TEXT,
            filename TEXT, mimetype TEXT, content BLOB,
            created_at TEXT NOT NULL
        );
        """
    )
    db.commit()
    db.close()


def current_worker():
    """The worker remembered on this device (via the session cookie), or None."""
    wid = session.get("worker_id")
    if not wid:
        return None
    return get_db().execute("SELECT * FROM workers WHERE id = ?", (wid,)).fetchone()


def cert_status(expiry_date):
    """Return (label, css_class, days_left) for a certificate's expiry date."""
    if not expiry_date:
        return ("No expiry", "ok", None)
    try:
        exp = date.fromisoformat(expiry_date)
    except ValueError:
        return ("No expiry", "ok", None)
    days = (exp - date.today()).days
    if days < 0:
        return (f"Expired {-days} d ago", "expired", days)
    if days <= EXPIRY_WARNING_DAYS:
        return (f"Expires in {days} d", "expiring", days)
    return (f"Valid · expires {expiry_date}", "ok", days)


def worker_certs(worker_id):
    """Certificate rows for a worker, each with computed expiry status."""
    rows = get_db().execute(
        "SELECT id, cert_type, cert_number, issued_date, expiry_date, filename, "
        "mimetype FROM certificates WHERE worker_id = ? ORDER BY cert_type",
        (worker_id,),
    ).fetchall()
    certs = []
    for r in rows:
        label, css, days = cert_status(r["expiry_date"])
        certs.append({**dict(r), "status": label, "status_class": css,
                      "days_left": days})
    return certs


# --- Admin: talks (same as the simple variant) ------------------------------


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if secrets.compare_digest(request.form.get("password", ""), ADMIN_PASSWORD):
            session.permanent = True
            session["is_admin"] = True
            nxt = request.args.get("next")
            return redirect(nxt if nxt and nxt.startswith("/") else url_for("index"))
        return render_template("login.html", error="Incorrect password.")
    return render_template("login.html")


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


@app.route("/")
@require_admin
def index():
    talks = get_db().execute(
        "SELECT t.*, COUNT(s.id) AS attendees FROM talks t "
        "LEFT JOIN signoffs s ON s.talk_id = t.id GROUP BY t.id ORDER BY t.id DESC"
    ).fetchall()
    return render_template("index.html", talks=talks)


@app.route("/talk/new", methods=["GET", "POST"])
@require_admin
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
            (topic, (request.form.get("presenter") or "").strip(),
             request.form.get("talk_date") or date.today().isoformat(),
             (request.form.get("key_points") or "").strip(),
             secrets.token_urlsafe(9),
             datetime.now().isoformat(timespec="seconds")),
        )
        db.commit()
        return redirect(url_for("talk_detail", talk_id=cur.lastrowid))
    return render_template("new_talk.html", today=date.today().isoformat())


@app.route("/talk/<int:talk_id>")
@require_admin
def talk_detail(talk_id):
    db = get_db()
    talk = db.execute("SELECT * FROM talks WHERE id = ?", (talk_id,)).fetchone()
    if talk is None:
        abort(404)
    signoffs = db.execute(
        "SELECT name, signed_at FROM signoffs WHERE talk_id = ? ORDER BY id DESC",
        (talk_id,),
    ).fetchall()
    return render_template("talk_detail.html", talk=talk, signoffs=signoffs)


@app.route("/talk/<int:talk_id>/qr.svg")
@require_admin
def talk_qr(talk_id):
    talk = get_db().execute("SELECT token FROM talks WHERE id = ?",
                            (talk_id,)).fetchone()
    if talk is None:
        abort(404)
    url = url_for("attend", token=talk["token"], _external=True)
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgImage)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), mimetype="image/svg+xml")


@app.route("/talk/<int:talk_id>/export.csv")
@require_admin
def export_csv(talk_id):
    db = get_db()
    talk = db.execute("SELECT * FROM talks WHERE id = ?", (talk_id,)).fetchone()
    if talk is None:
        abort(404)
    rows = db.execute(
        "SELECT name, signed_at, ip FROM signoffs WHERE talk_id = ? ORDER BY id",
        (talk_id,),
    ).fetchall()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([f"Toolbox talk: {talk['topic']}"])
    w.writerow([f"Presenter: {talk['presenter']}  Date: {talk['talk_date']}"])
    w.writerow([])
    w.writerow(["Name", "Signed off (biometric verified)", "Signed at", "IP"])
    for r in rows:
        w.writerow([r["name"], "Yes", r["signed_at"], r["ip"]])
    safe = "".join(c if c.isalnum() else "_" for c in talk["topic"])[:40]
    return Response(out.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename=toolbox_{safe}_{talk['talk_date']}.csv"})


# --- Worker: scan -> (enrol once) -> sign off with fingerprint --------------


@app.route("/attend/<token>")
def attend(token):
    talk = get_db().execute("SELECT * FROM talks WHERE token = ?",
                            (token,)).fetchone()
    if talk is None:
        abort(404)
    worker = current_worker()
    already = None
    if worker:
        row = get_db().execute(
            "SELECT signed_at FROM signoffs WHERE talk_id = ? AND worker_id = ?",
            (talk["id"], worker["id"]),
        ).fetchone()
        already = row["signed_at"] if row else None
    return render_template("attend.html", talk=talk, worker=worker, already=already)


@app.route("/enroll/options", methods=["POST"])
def enroll_options():
    name = (request.get_json(silent=True) or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "Please enter your name."}), 400
    # A fresh random user handle for this enrolment (created as a worker on verify).
    user_handle = secrets.token_bytes(16)
    options = generate_registration_options(
        rp_id=RP_ID,
        rp_name=RP_NAME,
        user_name=name,
        user_id=user_handle,
        user_display_name=name,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.REQUIRED,
        ),
    )
    session["reg_challenge"] = bytes_to_base64url(options.challenge)
    session["reg_name"] = name
    return Response(options_to_json(options), mimetype="application/json")


@app.route("/enroll/verify", methods=["POST"])
def enroll_verify():
    challenge = session.pop("reg_challenge", None)
    name = session.pop("reg_name", None)
    if not challenge or not name:
        return jsonify({"error": "Enrolment expired, please start again."}), 400
    try:
        verification = verify_registration_response(
            credential=request.get_data(as_text=True),
            expected_challenge=base64url_to_bytes(challenge),
            expected_rp_id=RP_ID,
            expected_origin=ORIGIN,
            require_user_verification=True,
        )
    except Exception as e:  # noqa: BLE001 - report any verification failure to client
        return jsonify({"error": f"Could not verify passkey: {e}"}), 400

    db = get_db()
    now = datetime.now().isoformat(timespec="seconds")
    cur = db.execute("INSERT INTO workers (name, created_at) VALUES (?, ?)",
                     (name, now))
    worker_id = cur.lastrowid
    db.execute(
        "INSERT INTO credentials (cred_id, worker_id, public_key, sign_count, "
        "created_at) VALUES (?, ?, ?, ?, ?)",
        (bytes_to_base64url(verification.credential_id), worker_id,
         bytes_to_base64url(verification.credential_public_key),
         verification.sign_count, now),
    )
    db.commit()
    session.permanent = True
    session["worker_id"] = worker_id
    return jsonify({"ok": True, "name": name})


@app.route("/attend/<token>/auth/options", methods=["POST"])
def auth_options(token):
    talk = get_db().execute("SELECT id FROM talks WHERE token = ?",
                            (token,)).fetchone()
    if talk is None:
        abort(404)
    worker = current_worker()
    if not worker:
        return jsonify({"error": "This device isn't set up yet."}), 400
    creds = get_db().execute(
        "SELECT cred_id FROM credentials WHERE worker_id = ?", (worker["id"],)
    ).fetchall()
    options = generate_authentication_options(
        rp_id=RP_ID,
        allow_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["cred_id"]))
            for c in creds
        ],
        user_verification=UserVerificationRequirement.REQUIRED,
    )
    session["auth_challenge"] = bytes_to_base64url(options.challenge)
    return Response(options_to_json(options), mimetype="application/json")


@app.route("/attend/<token>/auth/verify", methods=["POST"])
def auth_verify(token):
    db = get_db()
    talk = db.execute("SELECT * FROM talks WHERE token = ?", (token,)).fetchone()
    if talk is None:
        abort(404)
    worker = current_worker()
    challenge = session.pop("auth_challenge", None)
    if not worker or not challenge:
        return jsonify({"error": "Sign-off expired, please try again."}), 400

    body = request.get_json(silent=True) or {}
    cred_row = db.execute(
        "SELECT * FROM credentials WHERE cred_id = ? AND worker_id = ?",
        (body.get("id"), worker["id"]),
    ).fetchone()
    if cred_row is None:
        return jsonify({"error": "Unknown passkey for this worker."}), 400

    try:
        verification = verify_authentication_response(
            credential=request.get_data(as_text=True),
            expected_challenge=base64url_to_bytes(challenge),
            expected_rp_id=RP_ID,
            expected_origin=ORIGIN,
            credential_public_key=base64url_to_bytes(cred_row["public_key"]),
            credential_current_sign_count=cred_row["sign_count"],
            require_user_verification=True,
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Verification failed: {e}"}), 400

    db.execute("UPDATE credentials SET sign_count = ? WHERE cred_id = ?",
               (verification.new_sign_count, cred_row["cred_id"]))

    # Record once per worker per talk.
    existing = db.execute(
        "SELECT signed_at FROM signoffs WHERE talk_id = ? AND worker_id = ?",
        (talk["id"], worker["id"]),
    ).fetchone()
    if existing:
        db.commit()
        return jsonify({"ok": True, "already": True, "name": worker["name"],
                        "time": existing["signed_at"]})

    signed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute(
        "INSERT INTO signoffs (talk_id, worker_id, name, signed_at, ip) "
        "VALUES (?, ?, ?, ?, ?)",
        (talk["id"], worker["id"], worker["name"], signed_at, request.remote_addr),
    )
    db.commit()
    return jsonify({"ok": True, "name": worker["name"], "time": signed_at})


# --- Certificates: worker wallet + admin overview ---------------------------


@app.route("/me")
def me():
    """A worker's own page: their qualifications and a form to add more."""
    worker = current_worker()
    certs = worker_certs(worker["id"]) if worker else []
    return render_template("me.html", worker=worker, certs=certs,
                           cert_types=CERT_TYPES)


@app.route("/me/certificates", methods=["POST"])
def add_certificate():
    worker = current_worker()
    if not worker:
        abort(403)
    cert_type = (request.form.get("cert_type") or "").strip()
    if cert_type == "Other":
        cert_type = (request.form.get("cert_type_other") or "").strip() or "Other"
    if not cert_type:
        return redirect(url_for("me"))

    filename = mimetype = content = None
    upload = request.files.get("file")
    if upload and upload.filename:
        mimetype = upload.mimetype
        if mimetype not in ALLOWED_CERT_MIME:
            return render_template(
                "me.html", worker=worker, certs=worker_certs(worker["id"]),
                cert_types=CERT_TYPES,
                error="File must be a PNG, JPG, WEBP image or a PDF.")
        content = upload.read()
        filename = upload.filename

    get_db().execute(
        "INSERT INTO certificates (worker_id, cert_type, cert_number, "
        "issued_date, expiry_date, filename, mimetype, content, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (worker["id"], cert_type,
         (request.form.get("cert_number") or "").strip(),
         request.form.get("issued_date") or None,
         request.form.get("expiry_date") or None,
         filename, mimetype, content,
         datetime.now().isoformat(timespec="seconds")),
    )
    get_db().commit()
    return redirect(url_for("me"))


@app.route("/me/certificates/<int:cert_id>/delete", methods=["POST"])
def delete_certificate(cert_id):
    worker = current_worker()
    if not worker:
        abort(403)
    get_db().execute("DELETE FROM certificates WHERE id = ? AND worker_id = ?",
                     (cert_id, worker["id"]))
    get_db().commit()
    return redirect(url_for("me"))


@app.route("/certificates/<int:cert_id>/file")
def certificate_file(cert_id):
    """Serve a stored certificate image/PDF (inline). Access: admin or owner."""
    row = get_db().execute(
        "SELECT worker_id, filename, mimetype, content FROM certificates "
        "WHERE id = ?", (cert_id,)
    ).fetchone()
    if row is None or row["content"] is None:
        abort(404)
    worker = current_worker()
    is_owner = worker and worker["id"] == row["worker_id"]
    if not (is_owner or session.get("is_admin")):
        abort(403)  # only the owning worker or a logged-in supervisor
    # Strict content type + nosniff so user uploads can't be treated as HTML.
    return Response(row["content"], mimetype=row["mimetype"], headers={
        "Content-Disposition": f'inline; filename="{row["filename"] or "certificate"}"',
        "X-Content-Type-Options": "nosniff",
    })


@app.route("/workers")
@require_admin
def workers():
    """Admin overview: every worker, their certs, and expiry warnings."""
    db = get_db()
    rows = db.execute("SELECT * FROM workers ORDER BY name").fetchall()
    people = []
    expiring_soon = []
    for w in rows:
        certs = worker_certs(w["id"])
        people.append({"id": w["id"], "name": w["name"], "certs": certs})
        for c in certs:
            if c["status_class"] in ("expiring", "expired"):
                expiring_soon.append({"name": w["name"], **c})
    expiring_soon.sort(key=lambda c: (c["days_left"] is None, c["days_left"]))
    return render_template("workers.html", people=people,
                           expiring_soon=expiring_soon)


@app.route("/forget", methods=["POST"])
def forget():
    """Forget the worker on this device (e.g. shared phone / re-enrol)."""
    session.pop("worker_id", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5008)
