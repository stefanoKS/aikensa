import cv2
import sys

def initialize_camera(): #Init 4k cam

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) #for ubuntu. It's D_SHOW for windows

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")


    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 134)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 128)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, 2000)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_GAMMA, 50)
    cap.set(cv2.CAP_PROP_GAIN, 100)

    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    # 4k res
    return cap
