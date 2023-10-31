import cv2
import numpy as np
import os
import datetime
from fpdf import FPDF

import cv2.aruco as aruco

dirname = os.path.dirname(__file__)

#generate A4 size charuco board 
dict_type = aruco.DICT_5X5_250
squares = (12, 8)
square_length = 0.020
marker_length = 0.017
aruco_dict = aruco.getPredefinedDictionary(dict_type)

LENGTH_PX = 1000  
MARGIN_PX=50




charboard = aruco.CharucoBoard(squares, square_length, marker_length, aruco_dict)
size_ratio = 210 / 297 #A4 size image

board = aruco.CharucoBoard.generateImage(charboard, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)


    
now = datetime.datetime.now()
filename = f"charuco_{now.strftime('%Y_%m_%f_%H%M%S')}.jpg"
pdfname = f"charucopdf_{now.strftime('%Y_%m_%f_%H%M%S')}.pdf"

filepath = os.path.join(dirname, filename)
pdfpath = os.path.join(dirname, pdfname)



cv2.imwrite(filepath, board)
print("png write to : ")
print(filepath)

pdf = FPDF(orientation='L', unit='mm', format='A4')
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=0)
#pdf.set_font("Arial", size=12)
pdf.image(filepath, x=0, y=0, w=297, h=210)
pdf.output(pdfpath)

print("pdf write to : ")
print(pdfpath)


cv2.imshow("board", board)
cv2.waitKey(3000)

# detector = aruco.CharucoDetector(board0)
# charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
# aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids, (255, 0, 0))

# board1 = aruco.CharucoBoard(squares, square_length_mm,
#     marker_length_mm, aruco_dict, np.arange(17) + 17) # 17 markers with ids  [17..33]
# detector = aruco.CharucoDetector(board1)
# charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
# aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids, (0, 255, 0))