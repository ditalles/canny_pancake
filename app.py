"""Flask server for the Pancake Printer system.

Serves the mobile web app and provides APIs for image processing,
G-code generation, payment, and printer control.
"""

import base64
import uuid

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image

import config
import payment
from gcode_generator import GCodeConfig, generate_gcode
from image_processor import (
    detect_and_crop_face,
    detect_edges,
    generate_preview,
    segment_tones,
)
from path_planner import plan_print_layers
from printer_driver import PrintManager

app = Flask(__name__)

# In-memory job storage (single-user RPi system)
jobs = {}

# Singleton print manager
print_manager = PrintManager()


def _image_to_base64(image, fmt=".png"):
    """Encode a cv2 image to base64 string."""
    _, buffer = cv2.imencode(fmt, image)
    return base64.b64encode(buffer).decode("utf-8")


def _decode_upload(file_storage):
    """Decode an uploaded file into a cv2 BGR image."""
    pil_image = Image.open(file_storage.stream)
    # Handle RGBA images
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# --- Page Routes ---


@app.route("/")
def index():
    return render_template(
        "index.html",
        stripe_key=payment.STRIPE_PUBLISHABLE_KEY,
        price_display=payment.get_price_display(),
        payment_enabled=payment.is_configured(),
    )


# --- Image Processing APIs ---


@app.route("/api/process", methods=["POST"])
def process_image():
    """Upload and process a portrait photo.

    Form fields:
        file: Image file
        method: Edge detection method (canny, auto_canny, log)
        blur_kernel: Blur kernel size (odd integer, default 5)
        low_threshold: Canny low threshold (default 100)
        high_threshold: Canny high threshold (default 200)
        num_tones: Number of tone levels (2 or 3, default 3)

    Returns:
        JSON with job_id, original_b64, edges_b64, tonal_preview_b64
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    method = request.form.get("method", "auto_canny")
    blur_kernel = int(request.form.get("blur_kernel", 5))
    low_threshold = int(request.form.get("low_threshold", 100))
    high_threshold = int(request.form.get("high_threshold", 200))
    num_tones = int(request.form.get("num_tones", config.DEFAULT_NUM_TONES))

    image = _decode_upload(file)
    face = detect_and_crop_face(image)
    edges = detect_edges(face, method, blur_kernel, low_threshold, high_threshold)
    tone_masks = segment_tones(face, num_tones)
    preview = generate_preview(face, edges, tone_masks)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "face": face,
        "edges": edges,
        "tone_masks": tone_masks,
        "num_tones": num_tones,
        "method": method,
        "blur_kernel": blur_kernel,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "gcode": None,
        "paid": not payment.is_configured(),  # Auto-approve if no payment configured
        "payment_intent_id": None,
    }

    return jsonify({
        "job_id": job_id,
        "original_b64": _image_to_base64(face),
        "edges_b64": _image_to_base64(edges),
        "tonal_preview_b64": _image_to_base64(preview),
        "num_tones": num_tones,
    })


@app.route("/api/reprocess", methods=["POST"])
def reprocess_image():
    """Re-run processing with adjusted parameters on an existing job.

    JSON body:
        job_id, method, blur_kernel, low_threshold, high_threshold, num_tones
    """
    data = request.get_json()
    job_id = data.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    method = data.get("method", job["method"])
    blur_kernel = int(data.get("blur_kernel", job["blur_kernel"]))
    low_threshold = int(data.get("low_threshold", job["low_threshold"]))
    high_threshold = int(data.get("high_threshold", job["high_threshold"]))
    num_tones = int(data.get("num_tones", job["num_tones"]))

    face = job["face"]
    edges = detect_edges(face, method, blur_kernel, low_threshold, high_threshold)
    tone_masks = segment_tones(face, num_tones)
    preview = generate_preview(face, edges, tone_masks)

    job.update({
        "edges": edges,
        "tone_masks": tone_masks,
        "num_tones": num_tones,
        "method": method,
        "blur_kernel": blur_kernel,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "gcode": None,
    })

    return jsonify({
        "edges_b64": _image_to_base64(edges),
        "tonal_preview_b64": _image_to_base64(preview),
        "num_tones": num_tones,
    })


# --- G-code APIs ---


@app.route("/api/generate-gcode", methods=["POST"])
def generate_gcode_endpoint():
    """Generate G-code from a processed job.

    JSON body:
        job_id: Required
        bed_width, bed_height, feed_dispense, extrusion_rate: Optional overrides
    """
    data = request.get_json()
    job_id = data.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    gcode_config = GCodeConfig(
        bed_width=float(data.get("bed_width", config.BED_WIDTH)),
        bed_height=float(data.get("bed_height", config.BED_HEIGHT)),
        feed_dispense=int(data.get("feed_dispense", config.FEED_RATE_DISPENSE)),
        extrusion_rate=float(data.get("extrusion_rate", config.EXTRUSION_RATE)),
    )

    layers = plan_print_layers(job["edges"], job["tone_masks"])
    image_shape = job["face"].shape
    gcode = generate_gcode(layers, image_shape, gcode_config)

    job["gcode"] = gcode
    gcode_lines = gcode.split("\n")

    return jsonify({
        "job_id": job_id,
        "total_lines": len(gcode_lines),
        "gcode_preview": "\n".join(gcode_lines[:50]),
    })


@app.route("/api/gcode/<job_id>")
def download_gcode(job_id):
    """Download the full G-code file for a job."""
    job = jobs.get(job_id)
    if not job or not job.get("gcode"):
        return jsonify({"error": "G-code not found"}), 404

    return Response(
        job["gcode"],
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename=pancake_{job_id[:8]}.gcode"},
    )


# --- Payment APIs ---


@app.route("/api/payment/create", methods=["POST"])
def create_payment():
    """Create a Stripe PaymentIntent for a pancake order.

    JSON body:
        job_id: Required
    """
    data = request.get_json()
    job_id = data.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    result = payment.create_payment_intent()
    if "error" in result:
        return jsonify(result), 500

    job["payment_intent_id"] = result["payment_intent_id"]
    return jsonify(result)


@app.route("/api/payment/verify", methods=["POST"])
def verify_payment_endpoint():
    """Verify that payment has been completed for a job.

    JSON body:
        job_id: Required
        payment_intent_id: Required
    """
    data = request.get_json()
    job_id = data.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    payment_intent_id = data.get("payment_intent_id")
    if payment.verify_payment(payment_intent_id):
        job["paid"] = True
        return jsonify({"verified": True})
    else:
        return jsonify({"verified": False, "error": "Payment not completed"}), 402


# --- Printer APIs ---


@app.route("/api/print", methods=["POST"])
def start_print():
    """Send a job to the printer.

    JSON body:
        job_id: Required

    Requires payment to be completed first (if payment is configured).
    """
    data = request.get_json()
    job_id = data.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if not job.get("gcode"):
        return jsonify({"error": "G-code not generated yet"}), 400

    if not job.get("paid"):
        return jsonify({"error": "Payment required before printing"}), 402

    print_job = print_manager.create_job(job["gcode"])
    job["print_job_id"] = print_job.id

    if not print_manager.start_job(print_job.id):
        return jsonify({"error": print_job.error_message or "Failed to start print"}), 500

    return jsonify({
        "job_id": job_id,
        "print_job_id": print_job.id,
        "status": "printing",
    })


@app.route("/api/status/<job_id>")
def get_status(job_id):
    """Poll the print status of a job."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    print_job_id = job.get("print_job_id")
    if not print_job_id:
        return jsonify({"status": "not_started", "progress": 0})

    print_job = print_manager.get_job(print_job_id)
    if not print_job:
        return jsonify({"error": "Print job not found"}), 404

    return jsonify({
        "status": print_job.status,
        "progress": round(print_job.progress, 1),
        "error_message": print_job.error_message,
    })


@app.route("/api/printer")
def printer_status():
    """Get printer connection status."""
    return jsonify(print_manager.printer_status)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=config.FLASK_DEBUG, port=config.FLASK_PORT)
