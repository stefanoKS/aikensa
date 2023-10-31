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

def detectAruco(image):
    return image
