import cv2
import numpy as np

# ---------------------------
# Translation
# ---------------------------
def translate_image(img, tx, ty):
    """Translate image by tx, ty pixels."""
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

# ---------------------------
# Scaling
# ---------------------------
def scale_image(img, sx, sy):
    """Scale image by sx, sy factors."""
    rows, cols = img.shape[:2]
    scaled = cv2.resize(img, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
    return scaled

# ---------------------------
# Rotation
# ---------------------------
def rotate_image(img, angle_deg, center=None):
    """Rotate image by angle_deg degrees."""
    rows, cols = img.shape[:2]
    if center is None:
        center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    return rotated