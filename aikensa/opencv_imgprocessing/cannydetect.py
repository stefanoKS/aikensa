import cv2
import numpy as np
import sys

def canny_edge_detection(image, opval, blval, lcval, ucval, conval, brval):

    # Apply adjustments
    
    adjusted_image = cv2.convertScaleAbs(image, alpha=conval, beta=0)
    brval /= 10  # More than 1 - brighter, less than 1 - darker
    lookup_table = np.array([((i / 255.0) ** brval) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(adjusted_image, lookup_table)

    gray_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blval | 1, blval | 1), 0)
    canny_edges = cv2.Canny(blurred_image, lcval, ucval)
    
    #canny_edges = cv2.bitwise_not(canny_edges)
    # Overlay canny edges on original image
    #overlay = cv2.addWeighted(image, opval, cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR), 1-opval, 0)
    overlay = cv2.addWeighted(image, opval, cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR), 1-opval, 0)
    
    return overlay