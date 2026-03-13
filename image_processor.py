"""Image processing pipeline: face detection, edge detection, tone segmentation."""

import cv2
import numpy as np


def detect_and_crop_face(image):
    """Detect a face using Haar cascades and return the cropped face region.

    Falls back to the original image if no face is detected.
    Adds padding around the face for a more natural crop.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return image

    # Take the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add 30% padding around the face for forehead/chin/ears
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

    Args:
        image: BGR input image
        method: "canny", "auto_canny", or "log"
        blur_kernel: Gaussian blur kernel size (must be odd)
        low_thresh: Canny low threshold (ignored for auto_canny/log)
        high_thresh: Canny high threshold (ignored for auto_canny/log)

    Returns:
        Binary edge image (uint8, 0 or 255)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure kernel size is odd
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

    Args:
        image: BGR input image
        num_tones: Number of tone levels (2 or 3)

    Returns:
        List of binary masks, ordered darkest-first.
        Each mask is uint8 with values 0 or 255.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute threshold boundaries evenly across the brightness range
    step = 256 // num_tones
    masks = []

    for i in range(num_tones):
        lower = i * step
        upper = 255 if i == num_tones - 1 else (i + 1) * step
        mask = cv2.inRange(gray, lower, upper)

        # Morphological cleanup: close small gaps, remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        masks.append(mask)

    return masks


def generate_preview(image, edge_image, tone_masks):
    """Generate a color-coded preview image showing tonal regions.

    Colors: darkest=red, mid=green, lightest=blue.
    Edge lines are overlaid in white.

    Args:
        image: Original BGR image
        edge_image: Binary edge image
        tone_masks: List of tone masks (darkest first)

    Returns:
        BGR preview image
    """
    preview = (image.copy() * 0.3).astype(np.uint8)  # Dim the original

    # Color overlays for each tone level
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

    # Overlay edges in white
    edge_color = np.zeros_like(preview)
    edge_color[:] = (255, 255, 255)
    edge_region = cv2.bitwise_and(edge_color, edge_color, mask=edge_image)
    preview = cv2.addWeighted(preview, 1.0, edge_region, 1.0, 0)

    return preview
