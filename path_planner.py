"""Path planning: contour extraction, simplification, optimization, and fill generation."""

import cv2
import numpy as np


def extract_contours(mask, min_area=50):
    """Extract external contours from a binary mask.

    Args:
        mask: Binary image (uint8, 0 or 255)
        min_area: Minimum contour area in pixels to keep (filters noise)

    Returns:
        List of contour arrays (each is Nx1x2 int32)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def simplify_contours(contours, epsilon_factor=0.002):
    """Reduce point count in contours using polygon approximation.

    Args:
        contours: List of contour arrays
        epsilon_factor: Approximation accuracy as fraction of perimeter

    Returns:
        List of simplified contour arrays
    """
    simplified = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) >= 2:
            simplified.append(approx)
    return simplified


def optimize_path(contours):
    """Reorder contours using nearest-neighbor to minimize travel distance.

    Starts from (0, 0) and greedily picks the contour whose start point
    is closest to the current position.

    Args:
        contours: List of contour arrays

    Returns:
        Reordered list of contour arrays
    """
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
        # Move current position to end of this contour
        current_pos = contours[best_idx][-1][0]

    return ordered


def create_fill_lines(mask, line_spacing_px=4):
    """Generate horizontal hatching lines to fill a masked region.

    Creates parallel horizontal line segments that cover the filled areas
    of the mask. This produces a visible density pattern for tonal regions.

    Args:
        mask: Binary image (uint8, 0 or 255)
        line_spacing_px: Pixel spacing between fill lines

    Returns:
        List of contour-like arrays (each is Nx1x2), one per fill line segment
    """
    h, w = mask.shape
    fill_lines = []

    for y in range(0, h, line_spacing_px):
        # Find runs of filled pixels on this row
        row = mask[y]
        in_run = False
        run_start = 0

        for x in range(w):
            if row[x] > 0 and not in_run:
                run_start = x
                in_run = True
            elif row[x] == 0 and in_run:
                if x - run_start > 2:  # Skip tiny segments
                    line = np.array([[[run_start, y]], [[x, y]]], dtype=np.int32)
                    fill_lines.append(line)
                in_run = False

        # Close any run that extends to the edge
        if in_run and w - run_start > 2:
            line = np.array([[[run_start, y]], [[w - 1, y]]], dtype=np.int32)
            fill_lines.append(line)

    return fill_lines


def plan_print_layers(edge_image, tone_masks, fill_spacing_px=4, min_contour_area=50):
    """Build the complete set of print layers from edges and tonal masks.

    Returns layers ordered darkest-first (dispense first = cook longest = darkest).

    Layer 0: Edge outlines (darkest batter, dispensed first)
    Layer 1: Dark tone fill
    Layer 2: Mid tone fill (if 3 tones)
    Light tone is typically not printed (it's the bare pancake color).

    Args:
        edge_image: Binary edge image
        tone_masks: List of tone masks (darkest first)
        fill_spacing_px: Pixel spacing for fill hatching
        min_contour_area: Minimum contour area to keep

    Returns:
        List of layers, each layer is a list of contour arrays
    """
    layers = []

    # Layer 0: Edge outlines
    edge_contours = extract_contours(edge_image, min_area=min_contour_area)
    edge_contours = simplify_contours(edge_contours)
    edge_contours = optimize_path(edge_contours)
    layers.append(edge_contours)

    # Subsequent layers: tonal fills (skip the lightest tone — that's bare pancake)
    tones_to_fill = tone_masks[:-1] if len(tone_masks) > 1 else tone_masks
    for mask in tones_to_fill:
        fill_lines = create_fill_lines(mask, line_spacing_px=fill_spacing_px)
        fill_lines = optimize_path(fill_lines)
        layers.append(fill_lines)

    return layers
