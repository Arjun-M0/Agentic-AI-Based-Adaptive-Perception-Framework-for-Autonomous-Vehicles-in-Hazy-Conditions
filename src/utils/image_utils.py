import cv2
import numpy as np

def calculate_illumination(image_bgr: np.ndarray) -> float:
    """
    Calculates the overall illumination (brightness) of an image.
    Uses the V (Value) channel of the HSV color space, normalized to 0.0 - 1.0.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
        
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # The 'Value' channel represents brightness
    v_channel = hsv[:, :, 2]
    
    # Calculate the mean brightness and normalize to 0.0 - 1.0
    mean_brightness = np.mean(v_channel) / 255.0
    return float(mean_brightness)

def calculate_contrast(image_bgr: np.ndarray) -> float:
    """
    Calculates the RMS (Root Mean Square) contrast of an image.
    Normalized based on standard deviation of grayscale intensities.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
        
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Standard deviation of pixel intensities is a standard measure of RMS contrast
    std_dev = np.std(gray)
    
    # Normalize (max standard deviation for an 8-bit image is roughly 127.5)
    normalized_contrast = min(std_dev / 127.5, 1.0)
    return float(normalized_contrast)
    
def calculate_haze_score(image_bgr: np.ndarray) -> float:
    """
    Estimates the haze severity using a simplified Dark Channel Prior (DCP).
    Higher values (closer to 1.0) indicate denser haze.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
        
    # The Dark Channel is the minimum intensity across all RGB channels for each pixel
    dark_channel = np.min(image_bgr, axis=2)
    
    # We apply a morphological erode (minimum filter) to represent local patches (e.g., 15x15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel_eroded = cv2.erode(dark_channel, kernel)
    
    # According to DCP theory, the mean of this dark channel strongly correlates with haze density
    haze_score = np.mean(dark_channel_eroded) / 255.0
    return float(haze_score)

def calculate_visibility_score(image_bgr: np.ndarray) -> float:
    """
    Estimates visibility based on high-frequency edge density.
    Haze obscures edges, so a lower edge density implies worse visibility.
    Scores closer to 1.0 indicate high visibility (clear edges).
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
        
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Use Sobel operators to find vertical and horizontal gradients (edges)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    magnitude = cv2.magnitude(sobelx, sobely)
    
    # The mean edge strength drops significantly in hazy/foggy conditions
    mean_edge_strength = np.mean(magnitude)
    
    # We scale it relative to an empirically determined baseline for clear images (e.g., ~50.0)
    # Capped at 1.0 to maintain the 0.0-1.0 boundary contract
    visibility_score = min(mean_edge_strength / 50.0, 1.0)
    return float(visibility_score)
