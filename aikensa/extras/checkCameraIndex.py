import cv2

def check_cameras():
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

if __name__ == "__main__":
    available_cameras = check_cameras()
    if available_cameras:
        print("Available camera indexes:", available_cameras)
    else:
        print("No cameras found.")
