# QR Attendance Check-in

Replace the paper attendance sheet with a single QR code at the door. People
scan it with their phone camera, confirm their name, and tap **Check in** — the
server records who was present and the exact time. No more passing a sheet
around and scribbling signatures.

## How it works

```
 [Printed QR poster]  --scan-->  phone opens /checkin  --tap name-->  saved to DB
                                                                         |
 Staff open /admin  <----- live list + CSV export <----------------------+
```

- **One shared QR code** is shown on the `/admin` page and can be printed.
- The link contains a secret `code` so people can't check in by guessing the URL.
- Records are stored in a local **SQLite** database (`attendance.db`).
- The admin page shows a live count + list and a **CSV export** for Excel.

## Setup

```bash
cd attendance
pip install -r requirements.txt
python app.py
```

Then on the machine running it:

- **Admin / QR code:** http://localhost:5006/admin
- People check in by scanning the QR (or visiting the `/checkin?code=...` link).

### Letting phones reach it

The app listens on `0.0.0.0:5006`, so any phone on the **same Wi-Fi** can reach
it at `http://<your-computer-ip>:5006`. For use outside your network, host it on
a small server / service (e.g. Render, Railway, a VPS) so the QR link works from
anywhere.

## Configuration (optional)

Set these as environment variables before launching:

| Variable                 | What it does                                              |
|--------------------------|----------------------------------------------------------|
| `ATTENDANCE_EVENT_CODE`  | Secret in the QR link. Change it + reprint to "rotate".   |
| `ATTENDANCE_ROSTER`      | Comma-separated names shown as quick-pick buttons.        |

Example:

```bash
ATTENDANCE_EVENT_CODE=mysecret123 \
ATTENDANCE_ROSTER="Daniel,Sam,Alex,Jordan" \
python app.py
```

## A note on "signatures"

The legal value of a signature on an attendance sheet is *proof a specific
person was present at a specific time*. A check-in with name + timestamp serves
the same purpose. If you need stronger proof you could later add: a code that
rotates every minute (so people can't check in from home), login via company
accounts, or photo capture at check-in.

## Files

- `app.py` — Flask server (routes, SQLite, QR generation, CSV export)
- `templates/checkin.html` — the page people see when they scan
- `templates/admin.html` — live attendance list + printable QR
- `requirements.txt` — dependencies
