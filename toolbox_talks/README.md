# Toolbox Talk Sign-off

Replace the paper sign-off sheet for safety **toolbox talks** with a QR code.
Workers scan, read the topic, and confirm "I attended and understood" — giving
you a timestamped, exportable compliance record for each talk.

## Why this matters

Toolbox talks (short safety briefings on sites and in factories) legally need a
record proving each worker **attended and understood** the topic. That paper
sheet everyone signs is what auditors and insurers ask for. This digitises it:
no lost sheets, no illegible signatures, instant export.

## How it works

```
 Supervisor creates a talk  -->  app generates a QR for THAT talk
        |                                   |
        |                            workers scan it
        v                                   v
 /talk/<id> dashboard  <--  read topic + tick "attended & understood" + name
   (live list + CSV)                        |
        ^------------------- signed, timestamped record ----+
```

- Each talk gets its **own unguessable QR link** (random token).
- Workers must **tick the acknowledgement box** — that's the legal "signature".
- Records are stored in **SQLite** (`toolbox.db`); export any talk to **CSV**.

## Setup

```bash
cd toolbox_talks
pip install -r requirements.txt
python app.py
```

- **Dashboard:** http://localhost:5007/  (create talks, view sign-offs)
- Workers sign by scanning each talk's QR (shown on its detail page).

The app listens on `0.0.0.0:5007`, so phones on the same Wi-Fi can reach it at
`http://<your-computer-ip>:5007`. To use it beyond your network, deploy it to a
small host so the QR links work from anywhere.

## Routes

| Route                     | Who    | Purpose                                  |
|---------------------------|--------|------------------------------------------|
| `/`                       | Admin  | List talks, create new ones              |
| `/talk/new`               | Admin  | Create a talk (topic, presenter, points) |
| `/talk/<id>`              | Admin  | QR code, live sign-off list, CSV export  |
| `/talk/<id>/qr.svg`       | Admin  | Printable QR (SVG)                       |
| `/talk/<id>/export.csv`   | Admin  | Download the sign-off record             |
| `/attend/<token>`         | Worker | Read the talk and sign off               |

## Possible next steps

- Admin login so only supervisors can create talks / see records.
- A reusable library of standard talk topics.
- Worker photo or drawn signature captured at sign-off, for extra proof.
- Auto-PDF of the signed register for filing.
