"""Test script to run the face detection + line drawing pipeline on a local image.

Usage:
    python test_pipeline.py photo.jpg
    python test_pipeline.py photo.jpg --method canny --blur 7 --tones 2
    python test_pipeline.py photo.jpg --output-dir results/

Saves output images to the current directory (or --output-dir):
    - 1_face_crop.png     : Detected and cropped face
    - 2_edges.png         : Line drawing (edge detection)
    - 3_tonal_preview.png : Color-coded tonal regions
    - 4_tone_0.png ...    : Individual tone masks
    - 5_gcode_preview.png : Visualization of G-code toolpaths
    - output.gcode        : Generated G-code file
"""

import argparse
import os
import sys

import cv2
import numpy as np

from image_processor import detect_and_crop_face, detect_edges, generate_preview, segment_tones
from path_planner import plan_print_layers
from gcode_generator import GCodeConfig, generate_gcode


def visualize_gcode(gcode_text, width=800, height=800):
    """Draw G-code toolpaths on a blank canvas for visual verification."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [
        (0, 0, 255),    # Red for T0 (darkest)
        (0, 200, 0),    # Green for T1
        (255, 150, 0),  # Blue for T2
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
                # Scale to canvas (assume 200mm bed)
                px = int(x / 200.0 * width)
                py = int((1.0 - y / 200.0) * height)  # Flip Y for display

                if is_dispensing and prev_x is not None:
                    ppx = int(prev_x / 200.0 * width)
                    ppy = int((1.0 - prev_y / 200.0) * height)
                    cv2.line(canvas, (ppx, ppy), (px, py), current_color, 1, cv2.LINE_AA)

                prev_x, prev_y = x, y

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Test pancake printer image pipeline")
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

    # Load image
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
