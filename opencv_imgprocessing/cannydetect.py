import cv2
import numpy as np
import sys

def canny_edge_detection(image, opval, blval, lcval, ucval, conval, brval):

    # Apply adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=conval, beta=brval)
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blval | 1, blval | 1), 0)
    canny_edges = cv2.Canny(blurred_image, lcval, ucval)
    
    # Overlay canny edges on original image
    overlay = cv2.addWeighted(image, opval, cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR), 1-opval, 0)
    
    return overlay