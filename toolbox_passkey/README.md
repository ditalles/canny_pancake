# Toolbox Talk Sign-off — Passkey edition 🔒

The "premium" variant: workers sign off a toolbox talk with their **fingerprint
/ Face ID / device PIN** instead of typing a name. After a one-time setup,
signing off is just **scan the QR → tap → fingerprint**.

## Why a passkey (and not "read the fingerprint")

A web app can **never** read a raw fingerprint — browsers don't expose it, for
privacy. The web standard for biometric login is **WebAuthn / passkeys**: the
phone uses the fingerprint/Face ID/PIN it already trusts to unlock an on-device
private key, and signs a one-time challenge from the server. The server checks
the signature. That proves *this enrolled person, on their device, just
authenticated with their biometric* — cryptographically stronger than a paper
signature, and the biometric never leaves the device.

## Flow

```
First time (once per worker, per device):
   scan QR → type name → "Set up with fingerprint" → phone biometric prompt
            → passkey created, device now remembers this worker

Every talk after that:
   scan QR → "Sign off as <name>" → fingerprint → done (no typing)
```

## ⚠️ Requirement: HTTPS (secure context)

WebAuthn only works on `https://` **or** on `http://localhost`. So:

- **Local testing** on the same machine: `http://localhost:5008` works.
- **Real use from phones** (over Wi-Fi / the internet): you **must** serve it
  over **HTTPS with a real hostname**, and set these env vars to match:

  ```bash
  export RP_ID=talks.example.com           # hostname only, no scheme/port
  export ORIGIN=https://talks.example.com  # full origin
  export FLASK_SECRET_KEY=$(python -c "import secrets;print(secrets.token_hex(32))")
  ```

  (Easiest paths to HTTPS: deploy behind a host like Render/Railway/Fly, or put
  Caddy/nginx with a Let's Encrypt cert in front.)

## Supervisor login

The dashboard, talk pages, CSV export and the `/workers` certificate overview
are behind a **password login** (`/admin/login`). Workers never log in — they
just scan a talk's QR code and use their own `/me` wallet.

Set the password via environment variable (the default `admin` is for local
testing only and prints a startup warning):

```bash
export ADMIN_PASSWORD="a-strong-password"
```

Because the login is a password, it must be served over **HTTPS** in
production (same requirement as the passkeys). A worker can always view their
own certificate file; only the owner or a logged-in supervisor can.

## Setup

```bash
cd toolbox_passkey
pip install -r requirements.txt
export ADMIN_PASSWORD="a-strong-password"   # omit for a default 'admin' (dev only)
python app.py
# open http://localhost:5008/  -> log in
```

## What's been tested vs. what needs a device

Verified here (server side): app boot, talk creation, QR, the WebAuthn
registration/authentication **options** endpoints, CSV export, and the guards
(empty name, signing off before enrolment, expired challenges).

**Needs a real phone + fingerprint over HTTPS to test end-to-end:** the actual
`navigator.credentials.create/get` biometric prompt and signature verification.
A headless server can't trigger a fingerprint sensor, so that final hop must be
checked on a device.

## Certificate wallet 🪪

Because the app already knows who each worker is, it doubles as a place to store
their **qualifications** — VCA, forklift/reach truck, working-at-height /
climbing, first aid, crane, etc. — with issue/expiry dates and a photo or PDF of
the certificate.

- **Workers** manage their own at `/me`: add a certificate (type, number, dates,
  optional file) and remove old ones.
- **Supervisors** see everyone at `/workers`, with a **"Needs attention"** panel
  that lists certificates that are **expired** or **expire within 60 days**
  (configurable) — so nobody operates a forklift on a lapsed ticket.
- Status is colour-coded: 🟢 valid · 🟠 expiring soon · 🔴 expired.

Files are limited to 8 MB and to PNG/JPG/WEBP/PDF, stored in the database, and
served with a strict content type + `nosniff` so an upload can't be treated as a
web page.

## Routes

| Route                              | Who    | Purpose                                |
|------------------------------------|--------|----------------------------------------|
| `/admin/login` · `/admin/logout`   | Admin  | Supervisor password login / logout     |
| `/`                                | Admin  | List / create talks; links to people   |
| `/talk/<id>`                       | Admin  | QR, live sign-offs, CSV export         |
| `/attend/<token>`                  | Worker | Enrol (first time) or sign off         |
| `/enroll/options` · `/enroll/verify`| Worker| WebAuthn registration (passkey setup)  |
| `/attend/<token>/auth/options` · `/auth/verify` | Worker | WebAuthn sign-off (fingerprint) |
| `/me`                              | Worker | My certificate wallet (add / remove)   |
| `/me/certificates`                 | Worker | Add a certificate (file upload)        |
| `/workers`                         | Admin  | All workers + certs + expiry warnings  |
| `/certificates/<id>/file`          | Owner/admin | View a stored certificate file    |
| `/forget`                          | Worker | Forget this device (shared phone)      |

## Data model

- `workers` — name + creation time
- `credentials` — each worker's passkey public key + signature counter
- `certificates` — per-worker qualifications: type, number, issue/expiry dates,
  and the certificate file (image/PDF)
- `talks` / `signoffs` — as in the simple variant; every sign-off here is
  biometric-verified

## Notes & next steps

- **Shared phones:** the "Not me? / Set up again" button forgets the device so
  the next person can enrol. For heavy shared use, consider the kiosk-badge
  model instead.
- **Admin login** gates all supervisor pages (set `ADMIN_PASSWORD`). For a
  larger team you'd likely want per-supervisor accounts rather than one shared
  password.
- **Lost phone / re-enrolment:** a worker simply enrols again on a new device;
  consider an admin screen to view/revoke a worker's passkeys.
