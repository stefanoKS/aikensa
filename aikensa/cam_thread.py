import cv2
import os
from datetime import datetime
import numpy as np
import yaml
import time

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from aikensa.camscripts.cam_init import initialize_camera
from PyQt5.QtGui import QImage, QPixmap
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix  
from aikensa.opencv_imgprocessing.cannydetect import canny_edge_detection
from aikensa.opencv_imgprocessing.detectaruco import detectAruco
from aikensa.opencv_imgprocessing.arucoplanarize import planarize
from aikensa.engine import create_inferer, EngineConfig, custom_infer_single

from aikensa.parts_config.cowltop_66832A030P import partcheck


from dataclasses import dataclass, field
from typing import List


@dataclass
class CameraConfig:
    opacity: float = 0.5
    blur: int = 10
    lower_canny: int = 100
    upper_canny: int = 200
    contrast: int = 200
    brightness: int = 0
    savecannyparams: bool = False
    capture: bool = False
    check_aruco: bool = False
    widget: int = 0
    calculatecamparams: bool = False
    
    rtinference: bool = False
    rtwarp: bool = True
    savewarp: bool = False
    delwarp: bool = False

    cannyreadwarp: bool = False

    delcamcalibration: bool = False

    #_________________________________________________________________________
    #_________________________________________________________________________
    #Cowl Top 6832A030P Param
    cowltoppitch: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])  # P1, P2, P3, P4, P5, Total Length
    cowltop_doInspect: bool = False
    #_________________________________________________________________________



class CameraThread(QThread):
    on_frame_raw = pyqtSignal(QImage)
    on_frame_processed = pyqtSignal(QImage)
    on_frame_aruco = pyqtSignal(QImage)
    on_inference = pyqtSignal(QImage)
    cowl_pitch_updated = pyqtSignal(list)

    def __init__(self, cam_config : CameraConfig = None, engine_config: EngineConfig = None, capture=False):
        super(CameraThread, self).__init__()
        self.running = True
        self.charucoTimer = None
        self.kensatimer = None

        if cam_config is None:
            self.cam_config = CameraConfig()
        else:
            self.cam_config = cam_config

        if engine_config is None:
            self.engine_config = EngineConfig()
        else:
            self.engine_config = engine_config
        self.inferer = create_inferer(self.engine_config)

    def run(self):
        cap = initialize_camera()
        while self.running:
            ret, raw_frame = cap.read()
            if ret:
                #print(self.cam_config.widget)
                #Read the /camcalibration, if it exists apply transformation to raw_frame
                #calibration_file_path = "./aikensa/cameracalibration/calibration_params.yaml"

                if os.path.exists("./aikensa/cameracalibration/calibration_params.yaml"):
                    with open("./aikensa/cameracalibration/calibration_params.yaml", 'r') as file:
                        calibration_data = yaml.load(file, Loader=yaml.FullLoader)
                        camera_matrix = np.array(calibration_data.get('camera_matrix'))
                        distortion_coefficients = np.array(calibration_data.get('distortion_coefficients'))
                        raw_frame = cv2.undistort(raw_frame, camera_matrix, distortion_coefficients, None, camera_matrix)
                    

                if self.cam_config.widget == 1:
                    current_time = time.time()
                    if self.cam_config.capture == True:
                        
                        raw_frame = detectCharucoBoard(raw_frame)
                        aruco_frame = raw_frame.copy()
                        self.charucoTimer = current_time

                        self.cam_config.capture = False

                    #override the raw_frame with aruco drawboard for 1 seconds
                    if self.charucoTimer and current_time - self.charucoTimer < 1:
                        raw_frame = aruco_frame

                    elif self.charucoTimer and current_time - self.charucoTimer >= 1:
                        self.charucoTimer = None

                    if self.cam_config.calculatecamparams == True:
                        calibration_matrix = calculatecameramatrix()
                        #Dump to "/calibration/calibration_params.yaml"
                        #Check if path exists
                        if not os.path.exists("./aikensa/cameracalibration"):
                            os.makedirs("./aikensa/cameracalibration")
                        with open("./aikensa/cameracalibration/calibration_params.yaml", "w") as file:
                            yaml.dump(calibration_matrix, file)

                        self.cam_config.calculatecamparams = False
                    if self.cam_config.delcamcalibration == True:
                        if os.path.exists("./aikensa/cameracalibration/calibration_params.yaml"):
                            os.remove("./aikensa/cameracalibration/calibration_params.yaml")
                            self.cam_config.delcamcalibration = False


                #Apply canny edge detection if widget is 2
                if self.cam_config.widget == 2:
                    planarized_canny=raw_frame.copy()
                    if self.cam_config.cannyreadwarp == True:
                        planarized_canny, _ = planarize(raw_frame)
                    processed_frame = canny_edge_detection(planarized_canny, self.cam_config.opacity, self.cam_config.blur, self.cam_config.lower_canny, self.cam_config.upper_canny, self.cam_config.contrast, self.cam_config.brightness)
                    #processed_frame = planarized_canny
                    qt_processed_frame = self.qt_processImage(processed_frame)
                    self.on_frame_processed.emit(qt_processed_frame)
                    if self.cam_config.capture == True:
                        if not os.path.exists("./aikensa/canny_image"):
                            os.makedirs("./aikensa/canny_image")

                        current_time = datetime.now().strftime("%y%m%d_%H%M%S")
                        file_name = f"canny{current_time}.png"
                        cv2.imwrite(os.path.join("./aikensa/canny_image", file_name), processed_frame)    
                        self.cam_config.capture = False

                    #print(self.cam_config.savecannyparams)
                    #save the blur, lowercanny, uppercanny, contrast, brightness to yaml file
                    if self.cam_config.savecannyparams == True:
                        params = {
                            'blur': self.cam_config.blur,
                            'lower_canny': self.cam_config.lower_canny,
                            'upper_canny': self.cam_config.upper_canny,
                            'contrast': self.cam_config.contrast,
                            'brightness': self.cam_config.brightness
                        }

                        if not os.path.exists("./aikensa/param"):
                            os.makedirs("./aikensa/param")

                        with open('./aikensa/param/cannyparams.yaml', 'w') as outfile:
                            yaml.dump(params, outfile, default_flow_style=False)

                        self.cam_config.savecannyparams = False


                if self.cam_config.widget == 3 and self.cam_config.capture == True:
                    
                    if not os.path.exists("./aikensa/training_image"):
                        os.makedirs("./aikensa/training_image")

                    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
                    file_name = f"capture_{current_time}.png"
                    cv2.imwrite(os.path.join("./aikensa/training_image", file_name), raw_frame)    
                    self.cam_config.capture = False
                    
                
                if self.cam_config.check_aruco == True and self.cam_config.widget == 4:
                    aruco_frame = detectAruco(raw_frame)
                    qt_aruco_frame = self.qt_processImage(aruco_frame)

                    self.on_frame_aruco.emit(qt_aruco_frame)


                #__________________________________________________________________________________________
                #__________________________________________________________________________________________
                #__________________________________________________________________________________________

                if self.cam_config.widget == 5: #-> CowlTop 66832A030P

                    planarized = raw_frame.copy()
                    current_time = time.time()

                    if self.cam_config.delwarp == True:
                        if os.path.exists("./aikensa/param/warptransform.yaml"):
                            os.remove("./aikensa/param/warptransform.yaml")
                        self.cam_config.delwarp = False


                    if self.cam_config.rtwarp:
                        planarized, transform = planarize(raw_frame)
                        
                        #save transform to yaml file if savewarp is true
                        

                        if self.cam_config.savewarp == True:
                            transform_list = transform.tolist()
                            if not os.path.exists("./aikensa/param"):
                                os.makedirs("./aikensa/param")

                            with open('./aikensa/param/warptransform.yaml', 'w') as outfile:
                                yaml.dump(transform_list, outfile, default_flow_style=False)

                            self.cam_config.savewarp = False

                    planarized_copy = planarized.copy() #copy for redrawing
                    qt_processed_frame = self.qt_processImage(planarized_copy, width=1791, height=591)

                    if self.cam_config.cowltop_doInspect == True:

                        self.kensatimer = current_time #initiate timer

                        detections, det_frame = custom_infer_single(self.inferer, planarized, self.engine_config.conf_thres, self.engine_config.iou_thres, self.engine_config.max_det)
                        imgcheck, pitch_results = partcheck(planarized_copy, detections)

                        if len(pitch_results) == len(self.cam_config.cowltoppitch):
                            self.cam_config.cowltoppitch = pitch_results

                        self.cam_config.cowltop_doInspect = False
                        imgresults = imgcheck.copy()

                    if self.kensatimer and current_time - self.kensatimer < 1.5:
                        qt_processed_frame = self.qt_processImage(imgresults, width=1791, height=591)

                    elif self.kensatimer and current_time - self.kensatimer >= 1.5:
                        qt_processed_frame = self.qt_processImage(planarized_copy, width=1791, height=591)
                        self.cam_config.cowltoppitch = [0,0,0,0,0,0]
                        self.kensatimer = None

                    self.on_inference.emit(qt_processed_frame)
                    self.cowl_pitch_updated.emit(self.cam_config.cowltoppitch)

                #__________________________________________________________________________________________
                #__________________________________________________________________________________________
                #__________________________________________________________________________________________



                    #change the qlabel color later
                    #self.on_inference.emit(qt_processed_frame)
                    


                qt_rawframe = self.qt_processImage(raw_frame)
                self.on_frame_raw.emit(qt_rawframe)
                
        cap.release()

    def stop(self):
        self.running = False

    def qt_processImage(self, image, width=1791, height=731):
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = processed_image.shape
        bytesPerLine = ch * w
        processed_image = QImage(processed_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        processed_image = processed_image.scaled(width, height, Qt.KeepAspectRatio)  
        
        return processed_image
    
    def loadCalibrationParams(calibration_file_path):
        with open(calibration_file_path, 'r') as file:
            calibration_data = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_data.get('camera_matrix'))
            distortion_coefficients = np.array(calibration_data.get('distortion_coefficients'))
        return camera_matrix, distortion_coefficients

    def _initialize_model(self):
        # Add button in corresponding widget then connect to this method
        return NotImplementedError()

    


