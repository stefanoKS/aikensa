import numpy as np
import cv2
import math
import yaml
import os
import pygame
import os
from PIL import ImageFont, ImageDraw, Image


pitchSpec = [11, 118, 98, 108, 25]
totalLengthSpec = 360
pitchTolerance = 1.5
totalLengthTolerance = 5.0

offset_y = 30 #offset for text and box
pixelMultiplier = 0.2488 #basically multiplier from 1/arucoplanarize param -> will create a constant for this later

endoffset_y = 0

kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

#Need to change the name of offset_y and endoffset_y -> too confusing

pygame.mixer.init()
ok_sound = pygame.mixer.Sound("aikensa/sound/positive_interface.wav") 
ng_sound = pygame.mixer.Sound("aikensa/sound/mixkit-classic-short-alarm-993.wav")  


def partcheck(img, detections):


    
    detections = sorted(detections, key=lambda x: x[1])

    leftmost_lengths = []
    middle_lengths = []
    rightmost_lengths = []

    detectedposX = []
    detectedposY = []
    detectedPosX = None
    detectedPosY = None

    prev_center = None
    flag_pitchfuryou = 0
    flag_clip_furyou = 0


    leftmost_detection = detections[0] if len(detections) > 0 else None 
    rightmost_detection = detections[-1] if len(detections) > 0 else None

    edge_left = None
    edge_right = None

    #flag the number of ana
    if len(detections) != 4:
        flag_clip_furyou = 1

    #use canny to check for left end pitch
    if leftmost_detection:
        leftmost_center_pixel = yolo_to_pixel(leftmost_detection, img.shape)
        edge_left = find_edge_point(img, leftmost_center_pixel, "left", offsetval = endoffset_y)
        # edge_left = find_edge_and_draw_line(img, leftmost_center_pixel, "left")
        if edge_left is not None:  # If an edge was found
            length_left = calclength(leftmost_center_pixel, edge_left)*pixelMultiplier
            leftmost_lengths.append(length_left)
            img = drawbox(img, edge_left, length_left)
            img = drawtext(img, edge_left, length_left)

    #use canny to check for right end pitch
    if rightmost_detection:
        rightmost_center_pixel = yolo_to_pixel(rightmost_detection, img.shape)
        edge_right = find_edge_point(img, rightmost_center_pixel, "right", offsetval = endoffset_y)
        # edge_right = find_edge_and_draw_line(img, rightmost_center_pixel, "right")
        if edge_right is not None:  # If an edge was found
            length_right = calclength(rightmost_center_pixel, edge_right)*pixelMultiplier
            rightmost_lengths.append(length_right)
            img = drawbox(img, edge_right, length_right)
            img = drawtext(img, edge_right, length_right)

    
    for detect in detections:
        class_id, x, y, w, h, confidence = detect
        center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]]) #draw bounding box also return center

        detectedposX.append(x*img.shape[1])
        detectedposY.append(y*img.shape[0])

        if prev_center is not None:
        # Draw line from the center of the previous bounding box to the current one
            cv2.line(img, prev_center, center, (0, 255, 0), 2)
            length = calclength(prev_center, center)*pixelMultiplier
            middle_lengths.append(length)
            # Calculate center of the line
            line_center = ((prev_center[0] + center[0]) // 2, (prev_center[1] + center[1]) // 2)
            img = drawbox(img, line_center, length)
            img = drawtext(img, line_center, length)
        prev_center = center

    detectedPitch = leftmost_lengths + middle_lengths + rightmost_lengths
    
    #Check if edge_left parameter exist
    if edge_left is not None:
        if edge_right is not None:
            detectedPosX = [edge_left[0]] + detectedposX + [edge_right[0]]
            detectedPosY = [edge_left[1]] + detectedposY + [edge_right[1]]
    else:
        detectedPosX = detectedposX
        detectedPosY = detectedposY
            
    total_length = sum(detectedPitch)

    pitchresult = check_tolerance(pitchSpec, totalLengthSpec, pitchTolerance, totalLengthTolerance, detectedPitch, total_length)

    if any(result != 1 for result in pitchresult):
        status = "NG"
        flag_pitchfuryou = 1
    else:
        status = "OK"

    xy_pairs = list(zip(detectedPosX, detectedPosY))
    draw_pitch_line(img, xy_pairs, pitchresult, endoffset_y)

    play_sound(status)
    img = draw_status_text(img, status)

    #draw flag in the left top corner
    img = draw_flag_status(img, flag_pitchfuryou, flag_clip_furyou)

    return img, pitchresult, detectedPitch, total_length

def play_sound(status):
    if status == "OK":
        ok_sound.play()
    elif status == "NG":
        ng_sound.play()

def draw_flag_status(image, flag_pitchfuryou, flag_clip_furyou):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(kanjiFontPath, 40)
    color=(200,10,10)
    if flag_pitchfuryou == 1:
        draw.text((120, 10), u"ピッチ不良", font=font, fill=color)  
    if flag_clip_furyou == 1:
        draw.text((120, 60), u"穴数不良", font=font, fill=color)  
    
    # Convert back to BGR for OpenCV compatibility
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return image


def draw_pitch_line(image, xy_pairs, pitchresult, endoffset_y):
    #cv2 works in int, so convert the xy_pairs to int
    xy_pairs = [(int(x), int(y)) for x, y in xy_pairs]
    for i in range(len(xy_pairs) - 1):

        if pitchresult[i] == 1:
            lineColor = (0, 255, 0)
        else:
            lineColor = (0, 0, 255)

        if i == 0:
            offsetpos_ = (xy_pairs[i+1][0], xy_pairs[i+1][1] - endoffset_y)
            cv2.line(image, xy_pairs[i], offsetpos_, lineColor, 2)
            #Draw small blue point at the xy_pairs
            cv2.circle(image, xy_pairs[i], 4, (255, 0, 0), -1)

        elif i == len(xy_pairs) - 2:
            offsetpos_ = (xy_pairs[i][0], xy_pairs[i][1] - endoffset_y)
            cv2.line(image, offsetpos_, xy_pairs[i+1], lineColor, 2)
            cv2.circle(image, xy_pairs[i+1], 4, (255, 0, 0), -1)


        else:
            cv2.line(image, xy_pairs[i], xy_pairs[i+1], lineColor, 2)
            # length = calclength(xy_pairs[i], xy_pairs[i+1])*pixelMultiplier 

    return None


#add "OK" and "NG"
def draw_status_text(image, status):
    # Define the position for the text: Center top of the image
    center_x = image.shape[1] // 2
    top_y = 50  # Adjust this value to change the vertical position

    # Text properties
    font_scale = 5.0  # Increased font scale for bigger text
    font_thickness = 8  # Increased font thickness for bolder text
    outline_thickness = font_thickness + 2  # Slightly thicker for the outline
    text_color = (0, 0, 255) if status == "NG" else (0, 255, 0)  # Red for NG, Green for OK
    outline_color = (0, 0, 0)  # Black for the outline

    # Calculate text size and position
    text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = center_x - text_size[0] // 2
    text_y = top_y + text_size[1]

    # Draw the outline
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, outline_thickness)

    # Draw the text over the outline
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    return image


def check_tolerance(pitchSpec, totalLengthSpec, pitchTolerance, totalLengthTolerance, detectedPitch, total_length):
    result = [0] * len(pitchSpec)
    
    for i, (spec, detected) in enumerate(zip(pitchSpec, detectedPitch)):
        if abs(spec - detected) <= pitchTolerance:
            result[i] = 1

    total_length_result = 1 if abs(totalLengthSpec - total_length) <= totalLengthTolerance else 0
    # print (totalLengthSpec, total_length, totalLengthTolerance)
    # print("Total Length Result: ", total_length_result)
    # Append the result for total length to the result array
    result.append(total_length_result)
    
    return result

def yolo_to_pixel(yolo_coords, img_shape):
    class_id, x, y, w, h, confidence = yolo_coords
    x_pixel = int(x * img_shape[1])
    y_pixel = int(y * img_shape[0])
    return x_pixel, y_pixel

# def find_edge_and_draw_line(image, center, direction="left"):
#     x, y = center[0], center[1]
#     blur = 0
#     brightness = 0
#     contrast = 1
#     lower_canny = 100
#     upper_canny = 200

#     #read canny value from /aikensa/param/canyparams.yaml if exist
#     if os.path.exists("./aikensa/param/cannyparams.yaml"):
#         with open("./aikensa/param/cannyparams.yaml") as f:
#             cannyparams = yaml.load(f, Loader=yaml.FullLoader)
#             blur = cannyparams["blur"]
#             brightness = cannyparams["brightness"]
#             contrast = cannyparams["contrast"]
#             lower_canny = cannyparams["lower_canny"]
#             upper_canny = cannyparams["upper_canny"]

#     # Apply adjustments
#     adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
#     gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
#     blurred_image = cv2.GaussianBlur(gray_image, (blur | 1, blur | 1), 0)
#     canny_img = cv2.Canny(blurred_image, lower_canny, upper_canny)

#     # cv2.imwrite("adjusted_image.jpg", adjusted_image)
#     # cv2.imwrite("gray_image.jpg", gray_image)
#     # cv2.imwrite("blurred_image.jpg", blurred_image)
#     # cv2.imwrite("canny_debug.jpg", canny_img)


#     while 0 <= x < image.shape[1]:
#         if canny_img[y, x] == 255:  # Found an edge
#             cv2.line(image, (center[0], center[1]), (x, y), (0, 255, 0), 1) #green color
#             color = (255, 0, 0)
#             cv2.circle(image, (x, y), 5, color, -1)
#             return x, y 
        
#         x = x - 1 if direction == "left" else x + 1

#     return None



def drawbox(image, pos, length):
    pos = (pos[0], pos[1] - offset_y)
    rectangle_bgr = (255, 255, 255)
    font_scale = 0.5
    font_thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(f"{length:.2f}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    top_left_x = pos[0] - text_width // 2 - 5
    top_left_y = pos[1] - text_height // 2 - 5
    bottom_right_x = pos[0] + text_width // 2 + 5
    bottom_right_y = pos[1] + text_height // 2 + 5
    
    cv2.rectangle(image, (top_left_x, top_left_y),
                  (bottom_right_x, bottom_right_y),
                  rectangle_bgr, -1)
    
    return image


def drawtext(image, pos, length):
    pos = (pos[0], pos[1] - offset_y)
    font_scale = 0.5
    font_thickness = 2
    text = f"{length:.1f}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    text_x = pos[0] - text_width // 2
    text_y = pos[1] + text_height // 2
    
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 125, 20), font_thickness)
    return image

def calclength(p1, p2):
    length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return length

def draw_bounding_box(image, x, y, w, h, img_size, color=(0, 255, 0), thickness=1):
    
    x = int(x * img_size[0])
    y = int(y * img_size[1])
    w = int(w * img_size[0])
    h = int(h * img_size[1])

    x1, y1 = int(x - w // 2), int(y - h // 2)
    x2, y2 = int(x + w // 2), int(y + h // 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    center_x, center_y = x, y
    return (center_x, center_y)

def find_edge_point(image, center, direction="None", offsetval = 0):
    x, y = center[0], center[1]
    blur = 0
    brightness = 0
    contrast = 1
    lower_canny = 100
    upper_canny = 200

    #read canny value from /aikensa/param/canyparams.yaml if exist
    if os.path.exists("./aikensa/param/cannyparams.yaml"):
        with open("./aikensa/param/cannyparams.yaml") as f:
            cannyparams = yaml.load(f, Loader=yaml.FullLoader)
            blur = cannyparams["blur"]
            brightness = cannyparams["brightness"]
            contrast = cannyparams["contrast"]
            lower_canny = cannyparams["lower_canny"]
            upper_canny = cannyparams["upper_canny"]

    # Apply adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur | 1, blur | 1), 0)
    canny_img = cv2.Canny(blurred_image, lower_canny, upper_canny)

    # cv2.imwrite(f"adjusted_image_{direction}.jpg", adjusted_image)
    # cv2.imwrite(f"gray_image.jpg_{direction}", gray_image)
    # cv2.imwrite(f"blurred_image.jpg_{direction}", blurred_image)
    # cv2.imwrite(f"canny_debug.jpg_{direction}", canny_img)


    while 0 <= x < image.shape[1]:
        if canny_img[y + offsetval, x] == 255:  # Found an edge
            # cv2.line(image, (center[0], center[1] + offsetval), (x, y + offsetval), (0, 255, 0), 1)
            # color = (0, 0, 255) if direction == "left" else (255, 0, 0)
            # cv2.circle(image, (x, y + offsetval), 5, color, -1)
            return x, y + offsetval
        
        x = x - 1 if direction == "left" else x + 1

    return None
