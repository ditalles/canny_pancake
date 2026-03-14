# PancakePrint

Turn a portrait photo into a custom pancake. Upload a selfie, and PancakePrint detects the face, extracts edges and tonal regions, generates G-code, and drives a CNC pancake printer to dispense batter onto a griddle — producing an edible portrait.

## How It Works

```
Photo → Face detection → Edge detection → Tone segmentation → Path planning → G-code → Printer → Pancake
```

1. **Face detection** — Haar cascade localizes and crops the face from the photo.
2. **Edge detection** — Canny or Laplacian filter extracts outlines as a binary mask.
3. **Tone segmentation** — Brightness thresholds split the face into 2–3 shade layers.
4. **Path planning** — Contours are extracted, simplified, and ordered (nearest-neighbor). Fill regions get horizontal hatching lines.
5. **G-code generation** — Pixel paths are converted to millimetre coordinates. Each tone becomes a tool layer (T0 dark, T1 medium, T2 light) so the printer switches batter dispensers between passes.
6. **Printing** — G-code is streamed over serial to a Marlin/GRBL controller that moves the print head and dispenses batter.

## Project Structure

| File | Purpose |
|---|---|
| `image_processor.py` | Face detection, edge detection, tone segmentation |
| `path_planner.py` | Contour extraction, simplification, ordering, hatching fill |
| `gcode_generator.py` | Converts paths to Marlin-compatible G-code |
| `printer_driver.py` | Serial communication, job queue, progress tracking |
| `app.py` | Flask web server and REST API |
| `payment.py` | Stripe payment integration |
| `config.py` | Centralised settings (bed size, feed rates, serial port, etc.) |
| `templates/index.html` | Single-page web UI |

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: Flask, opencv-python-headless, numpy, Pillow, pyserial, stripe.

## Usage

```bash
python app.py
```

Opens on `http://localhost:5005`. The UI lets you upload a photo, tweak processing parameters, preview the result, optionally pay via Stripe, and send the job to the printer.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PRINTER_PORT` | `/dev/ttyUSB0` | Serial device |
| `PRINTER_BAUD` | `115200` | Baud rate |
| `BED_WIDTH` | `200` | Print bed width (mm) |
| `BED_HEIGHT` | `200` | Print bed height (mm) |
| `FEED_RATE_DISPENSE` | `600` | Dispense speed (mm/min) |
| `EXTRUSION_RATE` | `0.05` | Batter per mm of travel |
| `DEFAULT_NUM_TONES` | `3` | Tone levels (2 or 3) |
| `DRY_RUN` | `false` | Simulate without printer |
| `FLASK_PORT` | `5005` | Web server port |
| `STRIPE_SECRET_KEY` | — | Stripe secret key |
| `STRIPE_PUBLISHABLE_KEY` | — | Stripe publishable key |
| `PANCAKE_PRICE_CENTS` | `500` | Price in cents ($5.00) |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/process` | POST | Upload and process a portrait |
| `/api/reprocess` | POST | Re-process with different parameters |
| `/api/generate-gcode` | POST | Generate G-code from processed image |
| `/api/gcode/<job_id>` | GET | Download G-code file |
| `/api/payment/create` | POST | Create Stripe PaymentIntent |
| `/api/payment/verify` | POST | Verify payment |
| `/api/print` | POST | Send job to printer |
| `/api/status/<job_id>` | GET | Poll print progress |
| `/api/printer` | GET | Printer connection status |

## Hardware

- CNC frame with X/Y/Z stepper motors
- Controller board running Marlin or GRBL firmware (e.g. RAMPS 1.4, SKR Mini E3)
- 2–3 batter dispensers (peristaltic pumps or syringes) mapped to tool positions T0–T2
- Temperature-controlled griddle
- Host computer (Raspberry Pi or similar) connected via USB serial
