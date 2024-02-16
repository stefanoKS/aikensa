import cv2
import numpy as np
import os
import datetime
from fpdf import FPDF

# Initialize ArUco dictionary and parameters for Charuco board
dict_type = cv2.aruco.DICT_6X6_250
squares = (24, 16)
square_length = 0.020
marker_length = 0.017
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

# A4 Paper dimensions in mm are 210 x 297
# Converting these to points: 1 mm = 2.83464567 points
A4_WIDTH_PT = int(297 * 2.83464567)
A4_HEIGHT_PT = int(210 * 2.83464567)

# A4 Paper dimensions in mm are 210 x 297
# Converting these to points: 1 mm = 2.83464567 points
A3_WIDTH_PT = int(420 * 2.83464567)
A3_HEIGHT_PT = int(297 * 2.83464567)


# Define padding
PADDING_PT = int(5 * 2.83464567)  # 5mm padding

charboard = cv2.aruco.CharucoBoard(squares, square_length, marker_length, aruco_dict)

charuco_width = A3_WIDTH_PT - 2*PADDING_PT
charuco_height = A3_HEIGHT_PT - 2*PADDING_PT

resMultiplier = 20 #necessary to get better board result

board = charboard.generateImage((charuco_width*resMultiplier, charuco_height*resMultiplier))


board_with_padding = 255 * np.ones((A3_HEIGHT_PT*resMultiplier, A3_WIDTH_PT*resMultiplier), dtype=np.uint8)

y_offset = (board_with_padding.shape[0] - board.shape[0]) // 2
x_offset = (board_with_padding.shape[1] - board.shape[1]) // 2

print("board_with_padding shape:", board_with_padding.shape)
print("board shape:", board.shape)
print("x_offset:", x_offset)
print("y_offset:", y_offset)

board_with_padding[y_offset:y_offset+board.shape[0], x_offset:x_offset+board.shape[1]] = board

output_dir = os.path.join(os.getcwd(), "charucoboard")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save as PNG
now = datetime.datetime.now()
filename = f"charuco_{now.strftime('%Y_%m_%d_%H%M%S')}.png"
filepath = os.path.join(output_dir, filename)
cv2.imwrite(filepath, board_with_padding)
print("PNG written to:")
print(filepath)

# Save as PDF using FPDF
pdf = FPDF(orientation='L', unit='pt', format='A3')
pdf.add_page()
pdf.image(filepath, x = 0, y = 0, w = A3_WIDTH_PT, h = A3_HEIGHT_PT)
pdfname = f"charuco_{now.strftime('%Y_%m_%d_%H%M%S')}.pdf"
pdfpath = os.path.join(output_dir, pdfname)
pdf.output(pdfpath)
print("PDF written to:")
print(pdfpath)
