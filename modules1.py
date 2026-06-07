import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================================
# HISTOGRAM PROCESSING
# ========================================

def histogram_equalization(img):
    """Perform histogram equalization."""
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def histogram_matching(source, reference):
    """Match the histogram of source to reference."""
    src = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb)
    matched = src.copy()
    for i in range(3):
        src_hist, bins = np.histogram(src[:, :, i].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref[:, :, i].flatten(), 256, [0, 256])
        cdf_src = np.cumsum(src_hist) / np.sum(src_hist)
        cdf_ref = np.cumsum(ref_hist) / np.sum(ref_hist)
        mapping = np.interp(cdf_src, cdf_ref, np.arange(256))
        matched[:, :, i] = cv2.LUT(src[:, :, i], mapping.astype(np.uint8))
    return cv2.cvtColor(matched, cv2.COLOR_YCrCb2BGR)

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """Adjust brightness and contrast."""
    beta = brightness      # range [-100,100]
    alpha = 1 + contrast/100.0  # contrast factor [1.0,3.0]
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


# ========================================
# SPATIAL FILTERING
# ========================================

def mean_filter(img, ksize=5):
    """Apply mean (average) filter."""
    return cv2.blur(img, (ksize, ksize))

def median_filter(img, ksize=5):
    """Apply median filter."""
    return cv2.medianBlur(img, ksize)

def laplacian_sharpen(img):
    """Apply Laplacian sharpening filter."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(gray - lap)
    return sharpened

def high_pass_filter(img):
    """Apply simple high-pass filter."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


# ========================================
# FREQUENCY DOMAIN FILTERING
# ========================================

def fourier_filter(img, filter_type='low', radius=30, band=20):
    """Perform low-pass, high-pass, or band-pass Fourier filtering."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)

    if filter_type == 'low':
        cv2.circle(mask, (ccol, crow), radius, 1, -1)
    elif filter_type == 'high':
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 0, -1)
    elif filter_type == 'band':
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius + band, 1, -1)
        cv2.circle(mask, (ccol, crow), radius, 0, -1)

    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(np.clip(img_back, 0, 255))


# ========================================
# MAIN DEMONSTRATION
# ========================================

if __name__ == "__main__":
    # Load the image
    img = cv2.imread("car1.jpg")

    # Histogram processing
    hist_eq = histogram_equalization(img)
    # For histogram matching, use the same image as reference here (for demo)
    hist_match = histogram_matching(img, hist_eq)
    bright_contrast = adjust_brightness_contrast(img, brightness=40, contrast=30)

    # Spatial filtering
    mean = mean_filter(img)
    median = median_filter(img)
    lap_sharp = laplacian_sharpen(img)
    high_pass = high_pass_filter(img)

    # Frequency domain filtering
    low_pass = fourier_filter(img, 'low', radius=30)
    high_pass_fft = fourier_filter(img, 'high', radius=30)
    band_pass = fourier_filter(img, 'band', radius=20, band=30)

    # Display results
    titles = [
        'Original', 'Histogram Equalization', 'Histogram Matching', 
        'Brightness/Contrast', 'Mean Filter', 'Median Filter',
        'Laplacian Sharpen', 'High-Pass (Spatial)', 
        'Low-Pass (FFT)', 'High-Pass (FFT)', 'Band-Pass (FFT)'
    ]
    images = [
        img, hist_eq, hist_match, bright_contrast,
        mean, median, lap_sharp, high_pass,
        low_pass, high_pass_fft, band_pass
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
