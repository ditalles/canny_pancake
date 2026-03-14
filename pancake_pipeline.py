#!/usr/bin/env python3
"""
Pancake Printer Pipeline — Single Self-Contained Script

Converts a portrait photo into G-code for a CNC pancake printer.

Usage:
    python pancake_pipeline.py photo.jpg
    python pancake_pipeline.py photo.jpg --method canny --blur 7 --tones 2
    python pancake_pipeline.py photo.jpg --output-dir results/

Dependencies:
    pip install opencv-python-headless numpy

Outputs (saved to --output-dir, default current directory):
    1_face_crop.png     — Detected and cropped face
    2_edges.png         — Line drawing (edge detection)
    3_tonal_preview.png — Color-coded tonal regions
    4_tone_0.png ...    — Individual tone masks
    5_gcode_preview.png — Visualization of G-code toolpaths
    output.gcode        — Generated G-code file
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Image Processing
# ──────────────────────────────────────────────────────────────────────────────

def detect_and_crop_face(image):
    """Detect a face using Haar cascades and return the cropped face region.

    Falls back to the original image if no face is detected.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return image

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad_x = int(w * 0.3)
    pad_y = int(h * 0.3)
    img_h, img_w = image.shape[:2]

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return image[y1:y2, x1:x2]


def detect_edges(image, method="canny", blur_kernel=5, low_thresh=100, high_thresh=200):
    """Run edge detection on an image.

    Methods: "canny", "auto_canny", "log"
    Returns binary edge image (uint8, 0 or 255).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    if method == "canny":
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
    elif method == "auto_canny":
        median_val = np.median(blurred)
        sigma = 0.33
        auto_low = int(max(0, (1.0 - sigma) * median_val))
        auto_high = int(min(255, (1.0 + sigma) * median_val))
        edges = cv2.Canny(blurred, auto_low, auto_high)
    elif method == "log":
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        _, edges = cv2.threshold(np.absolute(laplacian), 10, 255, cv2.THRESH_BINARY)
        edges = edges.astype(np.uint8)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")

    return edges


def segment_tones(image, num_tones=3):
    """Segment an image into tonal regions based on brightness.

    Returns list of binary masks, ordered darkest-first.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    step = 256 // num_tones
    masks = []

    for i in range(num_tones):
        lower = i * step
        upper = 255 if i == num_tones - 1 else (i + 1) * step
        mask = cv2.inRange(gray, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        masks.append(mask)

    return masks


def generate_preview(image, edge_image, tone_masks):
    """Generate a color-coded preview showing tonal regions with edge overlay."""
    preview = (image.copy() * 0.3).astype(np.uint8)

    colors = [
        (0, 0, 200),   # Red for dark (dispensed first, cooks longest)
        (0, 180, 0),   # Green for mid
        (200, 100, 0), # Blue for light
    ]

    for i, mask in enumerate(tone_masks):
        if i < len(colors):
            color_overlay = np.zeros_like(preview)
            color_overlay[:] = colors[i]
            region = cv2.bitwise_and(color_overlay, color_overlay, mask=mask)
            preview = cv2.addWeighted(preview, 1.0, region, 0.5, 0)

    edge_color = np.zeros_like(preview)
    edge_color[:] = (255, 255, 255)
    edge_region = cv2.bitwise_and(edge_color, edge_color, mask=edge_image)
    preview = cv2.addWeighted(preview, 1.0, edge_region, 1.0, 0)

    return preview


# ──────────────────────────────────────────────────────────────────────────────
# Path Planning
# ──────────────────────────────────────────────────────────────────────────────

def extract_contours(mask, min_area=50):
    """Extract external contours from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def simplify_contours(contours, epsilon_factor=0.002):
    """Reduce point count using polygon approximation."""
    simplified = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) >= 2:
            simplified.append(approx)
    return simplified


def optimize_path(contours):
    """Reorder contours using nearest-neighbor to minimize travel distance."""
    if len(contours) <= 1:
        return contours

    remaining = list(range(len(contours)))
    ordered = []
    current_pos = np.array([0, 0])

    while remaining:
        best_idx = None
        best_dist = float("inf")

        for idx in remaining:
            start_pt = contours[idx][0][0]
            dist = np.linalg.norm(current_pos - start_pt)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        remaining.remove(best_idx)
        ordered.append(contours[best_idx])
        current_pos = contours[best_idx][-1][0]

    return ordered


def create_fill_lines(mask, line_spacing_px=4):
    """Generate horizontal hatching lines to fill a masked region."""
    h, w = mask.shape
    fill_lines = []

    for y in range(0, h, line_spacing_px):
        row = mask[y]
        in_run = False
        run_start = 0

        for x in range(w):
            if row[x] > 0 and not in_run:
                run_start = x
                in_run = True
            elif row[x] == 0 and in_run:
                if x - run_start > 2:
                    line = np.array([[[run_start, y]], [[x, y]]], dtype=np.int32)
                    fill_lines.append(line)
                in_run = False

        if in_run and w - run_start > 2:
            line = np.array([[[run_start, y]], [[w - 1, y]]], dtype=np.int32)
            fill_lines.append(line)

    return fill_lines


def plan_print_layers(edge_image, tone_masks, fill_spacing_px=4, min_contour_area=50):
    """Build complete print layers from edges and tonal masks.

    Layer 0: Edge outlines (darkest batter, dispensed first)
    Layer 1: Dark tone fill
    Layer 2: Mid tone fill (if 3 tones)
    Light tone is not printed (bare pancake color).
    """
    layers = []

    edge_contours = extract_contours(edge_image, min_area=min_contour_area)
    edge_contours = simplify_contours(edge_contours)
    edge_contours = optimize_path(edge_contours)
    layers.append(edge_contours)

    tones_to_fill = tone_masks[:-1] if len(tone_masks) > 1 else tone_masks
    for mask in tones_to_fill:
        fill_lines = create_fill_lines(mask, line_spacing_px=fill_spacing_px)
        fill_lines = optimize_path(fill_lines)
        layers.append(fill_lines)

    return layers


# ──────────────────────────────────────────────────────────────────────────────
# G-code Generation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GCodeConfig:
    """Configuration for G-code generation."""
    bed_width: float = 200.0
    bed_height: float = 200.0
    feed_travel: int = 3000
    feed_dispense: int = 600
    extrusion_rate: float = 0.05
    z_height: float = 1.0
    tool_codes: list = field(default_factory=lambda: ["T0", "T1", "T2"])


def pixels_to_mm(points, image_shape, config):
    """Convert pixel coordinates to mm on the print bed (preserves aspect ratio)."""
    img_h, img_w = image_shape[:2]

    scale_x = config.bed_width / img_w
    scale_y = config.bed_height / img_h
    scale = min(scale_x, scale_y)

    offset_x = (config.bed_width - img_w * scale) / 2.0
    offset_y = (config.bed_height - img_h * scale) / 2.0

    mm_points = np.zeros_like(points, dtype=float)
    mm_points[:, 0] = points[:, 0] * scale + offset_x
    mm_points[:, 1] = (img_h - points[:, 1]) * scale + offset_y

    return mm_points


def _distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_gcode(layers, image_shape, config=None):
    """Generate complete Marlin G-code from print layers."""
    if config is None:
        config = GCodeConfig()

    lines = []

    lines.append("; Pancake Printer G-code")
    lines.append("; Generated by PancakePrint")
    lines.append(f"; Layers: {len(layers)}")
    lines.append("")
    lines.append("G21 ; Metric units (mm)")
    lines.append("G90 ; Absolute positioning")
    lines.append("G28 ; Home all axes")
    lines.append(f"G0 Z{config.z_height:.1f} F{config.feed_travel} ; Set nozzle height")
    lines.append("")

    total_extrusion = 0.0

    for layer_idx, contours in enumerate(layers):
        if not contours:
            continue

        tool = config.tool_codes[layer_idx] if layer_idx < len(config.tool_codes) else config.tool_codes[-1]
        lines.append(f"; --- Layer {layer_idx} ({tool}) ---")
        lines.append(f"{tool} ; Select batter dispenser")
        lines.append("G92 E0 ; Reset extruder position")
        total_extrusion = 0.0
        lines.append("")

        for contour in contours:
            pts = contour.reshape(-1, 2)
            if len(pts) < 2:
                continue

            mm_pts = pixels_to_mm(pts, image_shape, config)

            start = mm_pts[0]
            lines.append(f"G0 X{start[0]:.2f} Y{start[1]:.2f} F{config.feed_travel}")

            for i in range(1, len(mm_pts)):
                dist = _distance(mm_pts[i - 1], mm_pts[i])
                total_extrusion += dist * config.extrusion_rate
                lines.append(
                    f"G1 X{mm_pts[i][0]:.2f} Y{mm_pts[i][1]:.2f} "
                    f"E{total_extrusion:.4f} F{config.feed_dispense}"
                )

            if len(mm_pts) >= 3:
                dist = _distance(mm_pts[-1], mm_pts[0])
                total_extrusion += dist * config.extrusion_rate
                lines.append(
                    f"G1 X{mm_pts[0][0]:.2f} Y{mm_pts[0][1]:.2f} "
                    f"E{total_extrusion:.4f} F{config.feed_dispense}"
                )

            lines.append("")

    lines.append("; --- End ---")
    lines.append("G0 Z10.0 F{} ; Raise nozzle".format(config.feed_travel))
    lines.append("G28 X Y ; Home X and Y")
    lines.append("M84 ; Disable motors")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# G-code Visualization
# ──────────────────────────────────────────────────────────────────────────────

def visualize_gcode(gcode_text, width=800, height=800):
    """Draw G-code toolpaths on a blank canvas for visual verification."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [
        (0, 0, 255),
        (0, 200, 0),
        (255, 150, 0),
    ]

    current_color = colors[0]
    prev_x, prev_y = None, None
    is_dispensing = False

    for line in gcode_text.split("\n"):
        line = line.split(";")[0].strip()
        if not line:
            continue

        if line.startswith("T"):
            idx = int(line[1]) if len(line) > 1 and line[1].isdigit() else 0
            current_color = colors[min(idx, len(colors) - 1)]

        if line.startswith("G0") or line.startswith("G1"):
            is_dispensing = line.startswith("G1")
            x, y = prev_x, prev_y
            for part in line.split():
                if part.startswith("X"):
                    x = float(part[1:])
                elif part.startswith("Y"):
                    y = float(part[1:])

            if x is not None and y is not None:
                px = int(x / 200.0 * width)
                py = int((1.0 - y / 200.0) * height)

                if is_dispensing and prev_x is not None:
                    ppx = int(prev_x / 200.0 * width)
                    ppy = int((1.0 - prev_y / 200.0) * height)
                    cv2.line(canvas, (ppx, ppy), (px, py), current_color, 1, cv2.LINE_AA)

                prev_x, prev_y = x, y

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a portrait photo into G-code for a pancake printer"
    )
    parser.add_argument("image", help="Path to input image (portrait/selfie)")
    parser.add_argument("--method", default="auto_canny", choices=["canny", "auto_canny", "log"],
                        help="Edge detection method (default: auto_canny)")
    parser.add_argument("--blur", type=int, default=5, help="Blur kernel size (default: 5)")
    parser.add_argument("--low", type=int, default=100, help="Canny low threshold (default: 100)")
    parser.add_argument("--high", type=int, default=200, help="Canny high threshold (default: 200)")
    parser.add_argument("--tones", type=int, default=3, choices=[2, 3],
                        help="Number of batter tones (default: 3)")
    parser.add_argument("--output-dir", default=".", help="Output directory (default: current dir)")
    parser.add_argument("--bed-width", type=float, default=200.0, help="Bed width in mm")
    parser.add_argument("--bed-height", type=float, default=200.0, help="Bed height in mm")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image: {args.image}")
        sys.exit(1)
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")

    # Step 1: Face detection
    face = detect_and_crop_face(image)
    out = os.path.join(args.output_dir, "1_face_crop.png")
    cv2.imwrite(out, face)
    print(f"[1/5] Face crop: {face.shape[1]}x{face.shape[0]} -> {out}")

    # Step 2: Edge detection
    edges = detect_edges(face, args.method, args.blur, args.low, args.high)
    out = os.path.join(args.output_dir, "2_edges.png")
    cv2.imwrite(out, edges)
    edge_count = np.count_nonzero(edges)
    print(f"[2/5] Edges ({args.method}): {edge_count} edge pixels -> {out}")

    # Step 3: Tone segmentation
    tone_masks = segment_tones(face, args.tones)
    for i, mask in enumerate(tone_masks):
        out = os.path.join(args.output_dir, f"4_tone_{i}.png")
        cv2.imwrite(out, mask)
    print(f"[3/5] Segmented into {len(tone_masks)} tones")

    # Step 4: Preview
    preview = generate_preview(face, edges, tone_masks)
    out = os.path.join(args.output_dir, "3_tonal_preview.png")
    cv2.imwrite(out, preview)
    print(f"[4/5] Tonal preview -> {out}")

    # Step 5: G-code
    layers = plan_print_layers(edges, tone_masks)
    total_contours = sum(len(layer) for layer in layers)
    config = GCodeConfig(bed_width=args.bed_width, bed_height=args.bed_height)
    gcode = generate_gcode(layers, face.shape, config)
    gcode_lines = len(gcode.split("\n"))

    out = os.path.join(args.output_dir, "output.gcode")
    with open(out, "w") as f:
        f.write(gcode)
    print(f"[5/5] G-code: {total_contours} contours, {gcode_lines} lines -> {out}")

    # Bonus: G-code visualization
    gcode_vis = visualize_gcode(gcode)
    out = os.path.join(args.output_dir, "5_gcode_preview.png")
    cv2.imwrite(out, gcode_vis)
    print(f"      G-code visualization -> {out}")

    print(f"\nDone! Check {args.output_dir}/ for all output files.")
    print(f"Tip: Upload {os.path.join(args.output_dir, 'output.gcode')} to ncviewer.com for 3D path view")


if __name__ == "__main__":
    main()
