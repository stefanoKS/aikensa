import cv2
import numpy as np
import os
import yaml

#detect 4 aruco marker corner and make them into a planar rectangular with certain aspect ratio
#layout is id 0 for topleft, 1 for topright, 2 for bottomleft, 3 for bottomright

multiplier = 4.0
IMAGE_HEIGHT = int(137 * multiplier)
IMAGE_WIDTH = int(410 * multiplier)

dict_type = cv2.aruco.DICT_6X6_250
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
desired_plane = np.array([[0, 0], [IMAGE_WIDTH, 0], [0, IMAGE_HEIGHT], [IMAGE_WIDTH, IMAGE_HEIGHT]], dtype='float32')


def planarize(image):
    transform = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if os.path.exists("./aikensa/param/warptransform.yaml"):
        with open('./aikensa/param/warptransform.yaml', 'r') as file:
            transform_list = yaml.load(file, Loader=yaml.FullLoader)
            transform = np.array(transform_list)
        image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return image, None

    else:
        corners, ids, rejected = detector.detectMarkers(gray)
        if corners and ids is not None:

            top_left_corner = None
            top_right_corner = None
            bottom_left_corner = None
            bottom_right_corner = None

            for i, corner in zip(ids, corners):
                marker_id = i[0]
                if marker_id == 0:
                    # Top left corner of marker 0
                    top_left_corner = corner[0][0]
                elif marker_id == 1:
                    # Top right corner of marker 1
                    top_right_corner = corner[0][1]
                elif marker_id == 2:
                    # Bottom left corner of marker 2
                    bottom_left_corner = corner[0][3]
                elif marker_id == 3:
                    # Bottom right corner of marker 3
                    bottom_right_corner = corner[0][2]

            if top_left_corner is not None and top_right_corner is not None \
            and bottom_left_corner is not None and bottom_right_corner is not None:
                # Concatenate the corners in the desired order
                ordered_corners = np.array([
                    top_left_corner, top_right_corner,
                    bottom_left_corner, bottom_right_corner
                ], dtype='float32')

                transform = cv2.getPerspectiveTransform(ordered_corners, desired_plane)
                image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT))

            return image, transform
        else:
            return image, None

















    # corners, ids, rejected = detector.detectMarkers(gray)
    # if corners and ids is not None:

    #     top_left_corner = None
    #     top_right_corner = None
    #     bottom_left_corner = None
    #     bottom_right_corner = None

    #     for i, corner in zip(ids, corners):
    #         marker_id = i[0]
    #         if marker_id == 0:
    #             # Top left corner of marker 0
    #             top_left_corner = corner[0][0]
    #         elif marker_id == 1:
    #             # Top right corner of marker 1
    #             top_right_corner = corner[0][1]
    #         elif marker_id == 2:
    #             # Bottom left corner of marker 2
    #             bottom_left_corner = corner[0][3]
    #         elif marker_id == 3:
    #             # Bottom right corner of marker 3
    #             bottom_right_corner = corner[0][2]

    #     if top_left_corner is not None and top_right_corner is not None \
    #     and bottom_left_corner is not None and bottom_right_corner is not None:
    #         # Concatenate the corners in the desired order
    #         ordered_corners = np.array([
    #             top_left_corner, top_right_corner,
    #             bottom_left_corner, bottom_right_corner
    #         ], dtype='float32')
    #         #if warptransform.yaml exists, load it and warp the image if not, calculate
    #         if os.path.exists("./aikensa/param/warptransform.yaml"):
    #             with open('./aikensa/param/warptransform.yaml', 'r') as file:
    #                 transform_list = yaml.load(file, Loader=yaml.FullLoader)
    #                 transform = np.array(transform_list)
    #             image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT))
    #         else:
    #             transform = cv2.getPerspectiveTransform(ordered_corners, desired_plane)
    #             image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT))

            

        
    #     return image, transform
    # else:
    #     return image, None