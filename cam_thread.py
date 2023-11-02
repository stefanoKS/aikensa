import cv2
import os
from datetime import datetime
import numpy as np
import yaml

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from camscripts.cam_init import initialize_camera
from PyQt5.QtGui import QImage, QPixmap
from opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix  
from opencv_imgprocessing.cannydetect import canny_edge_detection
from opencv_imgprocessing.detectaruco import detectAruco

from dataclasses import dataclass

@dataclass
class CameraConfig:
    opacity: float = 0.5
    blur: int = 10
    lower_canny: int = 100
    upper_canny: int = 200
    contrast: int = 200
    brightness: int = 0
    capture: bool = False
    check_aruco: bool = False
    widget: int = 0
    calculatecamparams: bool = False


class CameraThread(QThread):
    on_frame_raw = pyqtSignal(QImage)
    on_frame_processed = pyqtSignal(QImage)
    on_frame_aruco = pyqtSignal(QImage)

    def __init__(self, config : CameraConfig = None , capture=False):
        super(CameraThread, self).__init__()
        self.running = True


        if config is None:
            self.config = CameraConfig()
        else:
            self.config = config

    def run(self):
        cap = initialize_camera()
        while self.running:
            ret, raw_frame = cap.read()
            if ret:
                #print(self.config.widget)
                
                
                qt_rawframe = self.qt_processImage(raw_frame)

                #Read the /camcalibration, if it exists apply transformation to raw_frame

                if os.path.exists("./cameracalibration/calibration_params.yaml"):
                    with open("./cameracalibration/calibration_params.yaml", 'r') as file:
                        calibration_data = yaml.load(file, Loader=yaml.FullLoader)
                        camera_matrix = np.array(calibration_data.get('camera_matrix'))
                        distortion_coefficients = np.array(calibration_data.get('distortion_coefficients'))
                        raw_frame = cv2.undistort(raw_frame, camera_matrix, distortion_coefficients, None, camera_matrix)
                        print("matrix detected")

                if self.config.widget == 1:
                    if self.config.capture == "True":
                        detectCharucoBoard(raw_frame)
                        #print(self.config.capture)
                        self.config.capture = "False"

                    if self.config.calculatecamparams == "True":
                        calibration_matrix = calculatecameramatrix()
                        #Dump to "/calibration/calibration_params.yaml"
                        #Check if path exists
                        if not os.path.exists("./cameracalibration"):
                            os.makedirs("./cameracalibration")
                        print("Calibration matrix: ", calibration_matrix)
                        with open("./cameracalibration/calibration_params.yaml", "w") as file:
                            yaml.dump(calibration_matrix, file)

                        self.config.calculatecamparams = "False"


                #Apply canny edge detection if widget is 2
                if self.config.widget == 2:
                    processed_frame = canny_edge_detection(raw_frame, self.config.opacity, self.config.blur, self.config.lower_canny, self.config.upper_canny, self.config.contrast, self.config.brightness)
                    qt_processed_frame = self.qt_processImage(processed_frame)
                    self.on_frame_processed.emit(qt_processed_frame)
                    if self.config.capture == "True":
                        if not os.path.exists("./canny_image"):
                            os.makedirs("./canny_image")

                        current_time = datetime.now().strftime("%y%m%d_%H%M%S")
                        file_name = f"canny{current_time}.png"
                        cv2.imwrite(os.path.join("./canny_image", file_name), processed_frame)    
                        self.config.capture = "False"


                if self.config.widget == 3 and self.config.capture == "True":
                    
                    if not os.path.exists("./training_image"):
                        os.makedirs("./training_image")

                    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
                    file_name = f"capture_{current_time}.png"
                    cv2.imwrite(os.path.join("./training_image", file_name), raw_frame)    
                    self.config.capture = "False"
                
                if self.config.check_aruco == "True" and self.config.widget == 4:
                    aruco_frame = detectAruco(raw_frame)
                    qt_aruco_frame = self.qt_processImage(aruco_frame)

                    self.on_frame_aruco.emit(qt_aruco_frame)
                    

                
                self.on_frame_raw.emit(qt_rawframe)
                
        cap.release()

    def stop(self):
        self.running = False

    def qt_processImage(self, image):
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = processed_image.shape
        bytesPerLine = ch * w
        processed_image = QImage(processed_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        processed_image = processed_image.scaled(1791, 731, Qt.KeepAspectRatio)  
        
        return processed_image
    
    def loadCalibrationParams(calibration_file_path):
        with open(calibration_file_path, 'r') as file:
            calibration_data = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_data.get('camera_matrix'))
            distortion_coefficients = np.array(calibration_data.get('distortion_coefficients'))
        return camera_matrix, distortion_coefficients


