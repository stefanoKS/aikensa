import cv2

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 2275)
# cap.set(cv2.CAP_PROP_TEMPERATURE, 2500)
# cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)

# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    #cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4200) # Set manual white balance temperature to 4200K


    cv2.imshow('White Balance Control', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
