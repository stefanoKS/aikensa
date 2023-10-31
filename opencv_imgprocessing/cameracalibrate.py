import cv2
import numpy as np
import yaml
import os

dict_type = cv2.aruco.DICT_6X6_250
squares = (12, 8)
square_length = 0.020
marker_length = 0.017
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
charboard = cv2.aruco.CharucoBoard(squares, square_length, marker_length, aruco_dict)

allCharucoCorners = []
allCharucoIds = []
allImagePoints = []
allObjectPoints = []

def detectCharucoBoard(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    
    if ids is not None:
        _, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charboard)

        if charucoCorners is not None and charucoIds is not None:
            allCharucoCorners.append(charucoCorners)
            allCharucoIds.append(charucoIds)
            _, currentObjectPoints, currentImagePoints = charboard.matchImagePoints(charucoCorners, charucoIds)
            allObjectPoints.append(currentObjectPoints)
            allImagePoints.append(currentImagePoints)

def calculatecameramatrix(n_image):
    """
    Calibrate camera using the collected corners, IDs, and matched points.
    """
    cameraMatrix = np.zeros((3, 3))
    distCoeffs = np.zeros((5, 1))
    rvecs = []
    tvecs = []
    calibration_flags = 0  # You can adjust flags here if necessary
    _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        allObjectPoints, allImagePoints, n_image, cameraMatrix, distCoeffs, flags=calibration_flags
    )
    calibration_data = {
        'camera_matrix': cameraMatrix.tolist(),
        'distortion_coefficients': distCoeffs.tolist(),
        'rotation_vectors': [r.tolist() for r in rvecs],
        'translation_vectors': [t.tolist() for t in tvecs]
    }
    
    # Save to file
    if not os.path.exists('../calibration/'):
        os.makedirs('../calibration/')
    with open('../calibration/calibration_params.yaml', 'w') as file:
        yaml.dump(calibration_data, file)
    
    return calibration_data