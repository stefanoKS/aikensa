import cv2
import numpy as np
import yaml
import os

from dataclasses import dataclass


dict_type = cv2.aruco.DICT_6X6_250
squares = (12, 8)
square_length = 0.020
marker_length = 0.017
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
charboard = cv2.aruco.CharucoBoard(squares, square_length, marker_length, aruco_dict)
detector = cv2.aruco.CharucoDetector(charboard)



allCharucoCorners = []
allCharucoIds = []
allObjectPoints = []
allImagePoints = []
imageSize = None
calibration_image = 0

@dataclass
class FontConfig:
    font_face: int = 0
    font_scale: float = 0.5
    font_thickness: int = 1
    font_color: tuple = (255, 255, 255)
    font_position: tuple = (0, 0)




def detectCharucoBoard(image):
    global allCharucoCorners, allCharucoIds, allObjectPoints, allImagePoints, imageSize, calibration_image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    imageSize = (width, height)

    charucoCorners, charucoIds, markerCorners, markersIds = detector.detectBoard(gray)   
    
    if charucoCorners is not None and charucoIds is not None:
        print("Charuco board detected.")
        allCharucoCorners.append(charucoCorners)
        allCharucoIds.append(charucoIds)
        currentObjectPoints, currentImagePoints = charboard.matchImagePoints(charucoCorners, charucoIds)
        allObjectPoints.append(currentObjectPoints)
        allImagePoints.append(currentImagePoints)

        #add calibration_image to the top of the image
        

    #Lets draw the markers
    image = cv2.aruco.drawDetectedMarkers(image, markerCorners, markersIds)

    return image
    

def calculatecameramatrix():
    global allObjectPoints, allImagePoints, imageSize

    if not allObjectPoints or not allImagePoints:
        print("Insufficient data for calibration.")
        return

    cameraMatrix = np.zeros((3, 3))
    distCoeffs = np.zeros((5, 1))
    rvecs = []
    tvecs = []
    calibration_flags = 0  # You can adjust flags here if necessary

    _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        allObjectPoints, allImagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags=calibration_flags
    )

    calibration_data = {
        'camera_matrix': cameraMatrix.tolist(),
        'distortion_coefficients': distCoeffs.tolist(),
        'rotation_vectors': [r.tolist() for r in rvecs],
        'translation_vectors': [t.tolist() for t in tvecs]
    }
    print (calibration_data)
    return calibration_data