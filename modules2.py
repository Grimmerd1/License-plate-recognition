import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================================
# EDGE DETECTION
# ========================================

def sobel_edge(img):
    """Edge detection using Sobel operator."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(grad_x, grad_y)
    return np.uint8(np.clip(sobel, 0, 255))

def prewitt_edge(img):
    """Edge detection using Prewitt operator."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    grad_x = cv2.filter2D(gray, -1, kernelx)
    grad_y = cv2.filter2D(gray, -1, kernely)
    prewitt = cv2.magnitude(np.float32(grad_x), np.float32(grad_y))
    return np.uint8(np.clip(prewitt, 0, 255))

def canny_edge(img, low=100, high=200):
    """Edge detection using Canny operator."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)

def compare_edges(img):
    """Compare Sobel, Prewitt, and Canny edge detection results."""
    sobel = sobel_edge(img)
    prewitt = prewitt_edge(img)
    canny = canny_edge(img)

    titles = ['Original', 'Sobel', 'Prewitt', 'Canny']
    images = [img, sobel, prewitt, canny]

    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ========================================
# IMAGE SEGMENTATION
# ========================================
def otsu_threshold(img):
    """Segment using Otsu’s thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
def region_growing(img, seed=None, threshold=15):
    """Simple region growing segmentation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if seed is None:
        seed = (gray.shape[0] // 2, gray.shape[1] // 2)
    region = np.zeros_like(gray)
    stack = [seed]
    region_mean = gray[seed]
    region_pixels = 1

    while stack:
        x, y = stack.pop()
        region[x, y] = 255
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                xn, yn = x + dx, y + dy
                if (0 <= xn < gray.shape[0]) and (0 <= yn < gray.shape[1]) and region[xn, yn] == 0:
                    if abs(int(gray[xn, yn]) - int(region_mean)) < threshold:
                        region[xn, yn] = 255
                        stack.append((xn, yn))
                        region_mean = (region_mean * region_pixels + gray[xn, yn]) / (region_pixels + 1)
                        region_pixels += 1
    return region

def color_segmentation(img, color_space='HSV'):
    """Segment image by color in RGB or HSV space."""
    if color_space == 'HSV':
        converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 100])   # Example: light color range
        upper = np.array([179, 80, 255])
    else:  # RGB
        converted = img
        lower = np.array([100, 100, 100])
        upper = np.array([255, 255, 255])

    mask = cv2.inRange(converted, lower, upper)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    return segmented


# ========================================
# MORPHOLOGICAL OPERATIONS
# ========================================

def morphological_ops(img):
    """Apply dilation, erosion, opening, and closing."""
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    erosion = cv2.erode(img, kernel, iterations=1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return dilation, erosion, opening, closing


# ========================================
# MAIN DEMONSTRATION
# ========================================

if __name__ == "__main__":
    # Load image
    img = cv2.imread("car1.jpg")

    # ---- Edge Detection ----
    sobel = sobel_edge(img)
    prewitt = prewitt_edge(img)
    canny = canny_edge(img)

    # ---- Segmentation ----
    otsu = otsu_threshold(img)
    region = region_growing(img)
    color_hsv = color_segmentation(img, 'HSV')
    color_rgb = color_segmentation(img, 'RGB')

    # ---- Morphological Processing ----
    dilation, erosion, opening, closing = morphological_ops(otsu)

    # ---- Visualization ----
    titles = [
        'Original', 'Sobel', 'Prewitt', 'Canny',
        'Otsu Thresholding', 'Region Growing', 'Color Seg (HSV)', 'Color Seg (RGB)',
        'Dilation', 'Erosion', 'Opening', 'Closing'
    ]
    images = [
        img, sobel, prewitt, canny,
        otsu, region, color_hsv, color_rgb,
        dilation, erosion, opening, closing
    ]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(3, 4, i + 1)
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
