import cv2
import cv2.aruco as aruco
import numpy as np
import sys

def detect_charuco_corners(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Define the Charuco dictionary and board
    dict_type = aruco.DICT_5X5_250
    squares = (12, 8)
    square_length = 0.020
    marker_length = 0.017
    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    board = aruco.CharucoBoard(squares, square_length, marker_length, aruco_dict)

    # Detect Aruco markers
    corners, ids, _ = aruco.detectMarkers(image, aruco_dict)
    

    if ids is not None and len(ids) > 0:
        # Interpolate Charuco corners
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, image, board)
        
        aruco.drawDetectedCornersCharuco(image.copy(), charuco_corners, charuco_ids, (255,0,0))

        
        cv2.imshow("corners", image)
        cv2.waitKey(5000)

        return charuco_corners
    
    else:
        return None

def main(image1_path, image2_path):
    # Detect Charuco corners in both images
    corners1 = detect_charuco_corners(image1_path)
    corners2 = detect_charuco_corners(image2_path)

    
    
    # print ("corners1 : ")
    # print (corners1)
    # print ("corners2 : ")
    # print (corners2)
    
    # Compute the homography matrix
    if corners1 is not None and corners2 is not None:
        ret, H = cv2.findHomography(corners1, corners2, cv2.RANSAC)
        #print(H.shape)
        # Stitch the images together
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        stitched = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        stitched[0:img2.shape[0], 0:img2.shape[1]] = img2

        # Save the result
        cv2.imwrite("stitched.jpg", stitched)
        print("Stitched image saved as stitched.jpg")

    else:
        print("Error: Couldn't detect enough points to compute the homography")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <image1_path> <image2_path>")
    else:
        main(sys.argv[1], sys.argv[2])
