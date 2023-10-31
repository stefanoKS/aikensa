import cv2
import numpy as np
import sys
import datetime

def detect_charuco_corners(inputimage):
    # Load the image
    image = inputimage

    # Define the Charuco dictionary and board
    dict_type = cv2.aruco.DICT_5X5_250
    square_length = 0.020
    marker_length = 0.017
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    board = cv2.aruco.CharucoBoard_create(12, 8, square_length, marker_length, aruco_dict)

    # Detect Aruco markers
    (corners, ids, _) = cv2.aruco.detectMarkers(image, aruco_dict)
    charuco_corners = np.array([])
    charuco_ids = np.array([])

    if (ids is not None):
        # Interpolate Charuco corners
        (_, c_corners, c_ids) = cv2.aruco.interpolateCornersCharuco(corners,
                                                                    ids,
                                                                    image,
                                                                    board)
        # Any corner is detected
        if c_ids is not None:
            charuco_corners = c_corners[:, 0]
            charuco_ids = c_ids[:, 0]

        return charuco_corners, charuco_ids
    
    else:
        return None


def drawDetectedCornersCharuco_own(img, corners, ids):
    """
    Draw rectangles and IDs to the corners

    Parameters
    ----------
    img : numpy.array()
        Two dimensional image matrix. Image can be grayscale image or RGB image
        including 3 layers. Allowed shapes are (x, y, 1) or (x, y, 3).
    corners : numpy.array()
        Checkerboard corners.
    ids : numpy.array()
        Corners' IDs.
    """

    if ids.size > 0:
        rect_size = 5
        id_font = cv2.FONT_HERSHEY_SIMPLEX
        id_scale = 0.5
        id_color = (255, 255, 0)
        rect_thickness = 1

        # Draw rectangels and IDs
        for (corner, id) in zip(corners, ids):
            corner_x = int(corner[0])
            corner_y = int(corner[1])
            id_text = "Id: {}".format(str(id))
            id_coord = (corner_x + 2*rect_size, corner_y + 2*rect_size)
            cv2.rectangle(img, (corner_x - rect_size, corner_y - rect_size),
                        (corner_x + rect_size, corner_y + rect_size),
                        id_color, thickness=rect_thickness)
            cv2.putText(img, id_text, id_coord, id_font, id_scale, id_color)


def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result


def main(image1_path, image2_path):
    # Detect Charuco corners in both images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    charuco_corners1, charuco_ids1 = detect_charuco_corners(img1)
    charuco_corners2, charuco_ids2 = detect_charuco_corners(img2)



    #drawDetectedCornersCharuco_own(img1.copy, charuco_corners1, charuco_ids1)
    # cv2.imshow("test", img1)
    # cv2.waitKey(5000)
    #drawDetectedCornersCharuco_own(img2.copy, charuco_corners2, charuco_ids2)
    # cv2.imshow("test", img2)
    # cv2.waitKey(5000)

        

    # print ("corners1 : ")
    # print (corners1)
    # print ("corners2 : ")
    # print (corners2)
    
    # Compute the homography matrix
    if charuco_corners1 is not None and charuco_corners2 is not None:
        M, mask = cv2.findHomography(charuco_corners1, charuco_corners2, cv2.RANSAC, 5.0)

        print(M)
        # h,w = img1.shape[:-1]
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts,M)

        # result = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
        # alpha = 0.5
        # blended_image = cv2.addWeighted(result, alpha, img2, 1 - alpha, 0)


        results = warpTwoImages(img2, img1, M)

        cv2.imshow("test", results)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()


        now = datetime.datetime.now()
        filename = f"capture_{now.strftime('%Y_%m_%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, results)
        print(f"Stitched image saved as {filename}.jpg")



    else:
        print("Error: Couldn't detect enough points to compute the homography")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <image1_path> <image2_path>")
    else:
        main(sys.argv[1], sys.argv[2])
