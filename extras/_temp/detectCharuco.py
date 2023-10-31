import cv2
import numpy as np
import os
import datetime
from fpdf import FPDF
import sys

import cv2.aruco as aruco


#generate A4 size charuco board 



def detect_charuco_board(image_path):

    image = cv2.imread(image_path)

    dict_type = aruco.DICT_5X5_250
    squares = (12, 8)
    square_length = 0.020
    marker_length = 0.017
    aruco_dict = aruco.getPredefinedDictionary(dict_type)

    LENGTH_PX = 1000  
    MARGIN_PX=50

    charboard = aruco.CharucoBoard(squares, square_length, marker_length, aruco_dict)
    detector = aruco.CharucoDetector(charboard)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
    aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids, (255, 0, 0))
    #aruco.drawDetectedMarkers(image, charuco_corners)

    


    print(charuco_corners)
    print(charuco_ids)
    print(marker_corners)
    print(marker_ids)

    cv2.imshow(f'charuco ids with OpenCV {cv2.__version__}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py image.png/jpeg")
    else:
        image_path = sys.argv[1]
        detect_charuco_board(image_path)