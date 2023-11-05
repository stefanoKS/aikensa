import cv2
import numpy as np
import yaml
import os

dict_type = cv2.aruco.DICT_6X6_250
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def detectAruco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    return image
