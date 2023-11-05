import cv2
import os

# Initialize the dictionary and parameters for ArUco marker generation
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate and save ArUco markers with IDs 0, 1, 2, 3
for i in range(4):
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, i, 400, 10)

    if not os.path.exists("./aruco"):
        os.makedirs("./aruco")

    filename = f"./aruco/ArUco_Marker_{i}.png"

    cv2.imwrite(filename, marker_image)

