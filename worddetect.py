import cv2
import numpy as np
import pytesseract
from scipy.optimize import curve_fit

def bezier_curve(t, p0, p1, p2, p3):
    """Definition of Bezier curve"""
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3

def fit_bezier_curve(points):
    """Fit a Bezier curve to given points"""
    # Initial guess for control points: start point, end point, and two intermediate points
    p0_guess = points[0]
    p1_guess = points[-1]
    p2_guess = np.average(points, axis=0)
    p3_guess = np.average(points[::-1], axis=0)
    
    # Curve fitting
    popt, _ = curve_fit(bezier_curve, np.linspace(0, 1, len(points)), points, p0=[p0_guess, p1_guess, p2_guess, p3_guess])
    
    return popt

# Preprocessing and word detection using Tesseract OCR
def preprocess_and_detect_words(image_path):
    # Read the input image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations (erosion and dilation) for noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Apply histogram equalization for contrast enhancement
    equ = cv2.equalizeHist(gray)

    # Use pytesseract for text detection and recognition
    text = pytesseract.image_to_string(equ)

    # Split the recognized text into words based on whitespace
    words = text.split()

    return words

# Example usage:
image_path = './taklu.png'
detected_words = preprocess_and_detect_words(image_path)
print("Detected Words:", detected_words)

# For each detected word, you can further process it as needed
