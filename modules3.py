import cv2
import matplotlib.pyplot as plt
from geom_utils import translate_image, scale_image, rotate_image

# 1. Load image
img = cv2.imread("car1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Apply transformations
translated = translate_image(img, tx=30, ty=15)
scaled = scale_image(img, sx=1.5, sy=1.3)
rotated = rotate_image(img, angle_deg=15)

# 3. Create figure
plt.figure(figsize=(12, 6))
plt.suptitle("Result", fontsize=24, color="navy", fontweight="bold")

# --- Translation ---
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(translated)
plt.title("Translated")
plt.axis("off")

# --- Scaling ---
plt.subplot(2, 3, 3)
plt.imshow(scaled)
plt.title("Scaled")
plt.axis("off")

# --- Rotation ---
plt.subplot(2, 3, 5)
plt.imshow(rotated)
plt.title("Rotated")
plt.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
