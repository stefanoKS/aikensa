import cv2
import os
import datetime


path = os.getcwd()
dirname = os.path.dirname(path)
savefolder = "capture"
folder = os.path.join(dirname, savefolder)

if not os.path.exists(folder):
    os.makedirs(folder)

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# fcc = ["Y", "U", "F", "Y"]

cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 24)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
cap.set(cv2.CAP_PROP_FOURCC, fourcc)


cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 6500)
# cap.set(cv2.CAP_PROP_TEMPERATURE, 10)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -3.0)
cap.set(cv2.CAP_PROP_SETTINGS, 1)
print(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
print(cap.get(cv2.CAP_PROP_TEMPERATURE))
print(cap.get(cv2.CAP_PROP_AUTO_WB))
print(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
h = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
print(codec)
print("CAP_PROP_WB_TEMPERATURE: " + str(cv2.CAP_PROP_WB_TEMPERATURE))
print(cv2.CAP_PROP_TEMPERATURE)
print(cap.get(cv2.CAP_PROP_FPS))
print("cude check")
print(cv2.cuda.getCudaEnabledDeviceCount())


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    #cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4200) # Set manual white balance temperature to 4200K


    cv2.imshow('White Balance Control', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        now = datetime.datetime.now()
        file_name = f"capture_{now.strftime('%Y_%m_%f_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(folder, file_name), frame)
        print("img write to : ")
        print(os.path.join(folder, file_name))

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
