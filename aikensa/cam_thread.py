import cv2
import os
from datetime import datetime
import numpy as np
import yaml
import time
import csv

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from aikensa.camscripts.cam_init import initialize_camera
from PyQt5.QtGui import QImage, QPixmap
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.opencv_imgprocessing.cannydetect import canny_edge_detection
from aikensa.opencv_imgprocessing.detectaruco import detectAruco
from aikensa.opencv_imgprocessing.arucoplanarize import planarize
from aikensa.engine import create_inferer, EngineConfig, custom_infer_single

from aikensa.parts_config.cowltop_66832A030P import partcheck as partcheck_idx5
from aikensa.parts_config.hoodrrside_5902A5XX import partcheck as partcheck_idx6
#from aikensa.parts_config.hoodrrsideRH_5902A510 import partcheck as partcheck_idx7

from PIL import ImageFont, ImageDraw, Image

from dataclasses import dataclass, field
from typing import List, Tuple


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
    takeimage_readwarp: bool = False

    # _________________________________________________________________________
    # _________________________________________________________________________
    # Cowl Top 6832A030P Param
    # P1, P2, P3, P4, P5, Total Length
    cowltoppitch: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    cowltop_doInspect: bool = False
    cowltop_doReinspect: bool = False
    cowltop_numofPart: Tuple[int, int] = (0, 0)
    resetCounter: bool = False
    cowltop_last_inspection_outcome: bool = None
    cowltop_last_inspect_maxredo: bool = None
    kensainName: str = None
    cowltop_konpokazu: int = 50
    # _________________________________________________________________________
    # HOOD RR SIDE Param
    #5902A509 param LH
    rrsideLHpitch: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0])
    rrsideLHnumofPart: Tuple[int, int] = (0, 0)
    rrsideLH_resetCounter: bool = False
    #5902A510 param RH
    rrsideRHpitch: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0])
    rrsideRHnumofPart: Tuple[int, int] = (0, 0)
    rrsideRH_resetCounter: bool = False



class CameraThread(QThread):
    on_frame_raw = pyqtSignal(QImage)
    on_frame_processed = pyqtSignal(QImage)
    on_frame_aruco = pyqtSignal(QImage)
    on_inference = pyqtSignal(QImage)

    cowl_pitch_updated = pyqtSignal(list)
    rrsideLH_pitch_updated = pyqtSignal(list)
    rrsideRH_pitch_updated = pyqtSignal(list)

    cowl_numofPart_updated = pyqtSignal(tuple)
    rrsideLH_numofPart_updated = pyqtSignal(tuple)
    rrsideRH_numofPart_updated = pyqtSignal(tuple)



    inspection_delay = 1.5



    def __init__(self, cam_config: CameraConfig = None):
        super(CameraThread, self).__init__()
        self.running = True
        self.charucoTimer = None
        self.kensatimer = None

        if cam_config is None:
            self.cam_config = CameraConfig()
        else:
            self.cam_config = cam_config

        self.widget_dir_map = {
            5: "66832A030P",
            6: "5902A509",
            7: "5902A510",
        }

        self.kanjiFontPath = "./aikensa/font/NotoSansJP-ExtraBold.ttf"


    def run(self):
        cap = initialize_camera()
   
        while self.running:
            ret, raw_frame = cap.read()
            raw_frame = self.rotate_frame(raw_frame)
            if ret:
                # print(self.cam_config.widget)
                # Read the /camcalibration, if it exists apply transformation to raw_frame
                # calibration_file_path = "./aikensa/cameracalibration/calibration_params.yaml"

                if os.path.exists("./aikensa/cameracalibration/calibration_params.yaml"):
                    with open("./aikensa/cameracalibration/calibration_params.yaml", 'r') as file:
                        calibration_data = yaml.load(
                            file, Loader=yaml.FullLoader)
                        camera_matrix = np.array(
                            calibration_data.get('camera_matrix'))
                        distortion_coefficients = np.array(
                            calibration_data.get('distortion_coefficients'))
                        raw_frame = cv2.undistort(
                            raw_frame, camera_matrix, distortion_coefficients, None, camera_matrix)
                        
                planarized_image = raw_frame.copy()
                
                if self.cam_config.widget == 1:
                    self.caruco_check(raw_frame)

                # Apply canny edge detection if widget is 2
                if self.cam_config.widget == 2:
                    self.canny_detection(raw_frame)

                if self.cam_config.widget == 3 and self.cam_config.capture == True:
                    self.generate_training_image(raw_frame)

                if self.cam_config.check_aruco == True and self.cam_config.widget == 4:
                    aruco_frame = detectAruco(raw_frame)
                    qt_aruco_frame = self.qt_processImage(aruco_frame)

                    self.on_frame_aruco.emit(qt_aruco_frame)

                # __________________________________________________________________________________________
                # __________________________________________________________________________________________
                # __________________________________________________________________________________________

                if self.cam_config.widget == 5:
                    self.part_inspect(raw_frame, 5)

                # __________________________________________________________________________________________
                # __________________________________________________________________________________________
                # __________________________________________________________________________________________

                if self.cam_config.widget == 6:
                    self.part_inspect(raw_frame, 6)
                    


                if self.cam_config.widget == 7:
                    self.part_inspect(raw_frame, 7)



                qt_rawframe = self.qt_processImage(raw_frame)
                self.on_frame_raw.emit(qt_rawframe)

        cap.release()

    def caruco_check(self, raw_frame):
        current_time = time.time()
        if self.cam_config.capture == True:

            raw_frame = detectCharucoBoard(raw_frame)
            aruco_frame = raw_frame.copy()
            self.charucoTimer = current_time

            self.cam_config.capture = False

        # override the raw_frame with aruco drawboard for 1 seconds
        if self.charucoTimer and current_time - self.charucoTimer < 1:
            raw_frame = aruco_frame

        elif self.charucoTimer and current_time - self.charucoTimer >= 1:
            self.charucoTimer = None

        if self.cam_config.calculatecamparams == True:
            calibration_matrix = calculatecameramatrix()
            # Dump to "/calibration/calibration_params.yaml"
            # Check if path exists
            if not os.path.exists("./aikensa/cameracalibration"):
                os.makedirs("./aikensa/cameracalibration")
            with open("./aikensa/cameracalibration/calibration_params.yaml", "w") as file:
                yaml.dump(calibration_matrix, file)

            self.cam_config.calculatecamparams = False
        if self.cam_config.delcamcalibration == True:
            if os.path.exists("./aikensa/cameracalibration/calibration_params.yaml"):
                os.remove(
                    "./aikensa/cameracalibration/calibration_params.yaml")
                self.cam_config.delcamcalibration = False

    def generate_training_image(self, raw_frame):
        os.makedirs("./aikensa/training_image", exist_ok=True)

        planarized_image = raw_frame.copy()
        if self.cam_config.takeimage_readwarp == True:
            planarizedtraining_frame, _ = planarize(planarized_image)
        qt_processed_frame = self.qt_processImage(planarizedtraining_frame)
        current_time = datetime.now().strftime("%y%m%d_%H%M%S")
        file_name = f"capture_{current_time}.png"
        cv2.imwrite(os.path.join(
            "./aikensa/training_image", file_name), planarizedtraining_frame)
        self.cam_config.capture = False

    def canny_detection(self, raw_frame):
        planarized_canny = raw_frame.copy()
        if self.cam_config.cannyreadwarp == True:
            planarized_canny, _ = planarize(raw_frame)
        processed_frame = canny_edge_detection(planarized_canny, self.cam_config.opacity, self.cam_config.blur,
                                               self.cam_config.lower_canny, self.cam_config.upper_canny, self.cam_config.contrast, self.cam_config.brightness)
        # processed_frame = planarized_canny
        qt_processed_frame = self.qt_processImage(processed_frame)
        self.on_frame_processed.emit(qt_processed_frame)
        if self.cam_config.capture == True:
            if not os.path.exists("./aikensa/canny_image"):
                os.makedirs("./aikensa/canny_image")

            current_time = datetime.now().strftime("%y%m%d_%H%M%S")
            file_name = f"canny{current_time}.png"
            cv2.imwrite(os.path.join(
                "./aikensa/canny_image", file_name), processed_frame)
            self.cam_config.capture = False

        # print(self.cam_config.savecannyparams)
        # save the blur, lowercanny, uppercanny, contrast, brightness to yaml file
        if self.cam_config.savecannyparams == True:
            params = {
                'blur': self.cam_config.blur,
                'lower_canny': self.cam_config.lower_canny,
                'upper_canny': self.cam_config.upper_canny,
                'contrast': self.cam_config.contrast,
                'brightness': self.cam_config.brightness
            }

            os.makedirs("./aikensa/param", exist_ok=True)

            with open('./aikensa/param/cannyparams.yaml', 'w') as outfile:
                yaml.dump(params, outfile,
                          default_flow_style=False)

            self.cam_config.savecannyparams = False

    def part_inspect(self, raw_frame, widgetidx):
        planarized = raw_frame.copy()
        current_time = time.time()

        if self.cam_config.delwarp == True:
            if os.path.exists("./aikensa/param/warptransform.yaml"):
                os.remove("./aikensa/param/warptransform.yaml")
            self.cam_config.delwarp = False

        if self.cam_config.rtwarp:
            planarized, transform = planarize(raw_frame)

            # save transform to yaml file if savewarp is true

            if self.cam_config.savewarp == True:
                transform_list = transform.tolist()
                os.makedirs("./aikensa/param", exist_ok=True)

                with open('./aikensa/param/warptransform.yaml', 'w') as outfile:
                    yaml.dump(transform_list, outfile,
                              default_flow_style=False)

                self.cam_config.savewarp = False

        planarized_copy = planarized.copy()  # copy for redrawing
        qt_processed_frame = self.qt_processImage(
            planarized_copy, width=1791, height=591)
        

        if widgetidx == 5:
            ok_count, ng_count = self.cam_config.cowltop_numofPart

            if self.cam_config.resetCounter == True:
                ok_count = 0
                ng_count = 0
                self.cam_config.cowltop_numofPart = (
                    ok_count, ng_count)
                self.cam_config.resetCounter = False

            if self.kensatimer:
                if current_time - self.kensatimer < self.inspection_delay:
                    self.cam_config.cowltop_doInspect = False
                    self.cam_config.cowltop_doReinspect = False

        elif widgetidx == 6:
            ok_count, ng_count = self.cam_config.rrsideLHnumofPart

            if self.cam_config.resetCounter == True:
                ok_count = 0
                ng_count = 0
                self.cam_config.rrsideLHnumofPart = (
                    ok_count, ng_count)
                self.cam_config.resetCounter = False

            if self.kensatimer:
                if current_time - self.kensatimer < self.inspection_delay:
                    self.cam_config.cowltop_doInspect = False

        elif widgetidx == 7:
            ok_count, ng_count = self.cam_config.rrsideRHnumofPart

            if self.cam_config.resetCounter == True:
                ok_count = 0
                ng_count = 0
                self.cam_config.rrsideRHnumofPart = (
                    ok_count, ng_count)
                self.cam_config.resetCounter = False

            if self.kensatimer:
                if current_time - self.kensatimer < self.inspection_delay:
                    self.cam_config.cowltop_doInspect = False



        # Check if the inspection flag is True
        if self.cam_config.cowltop_doInspect == True:
            if self.kensatimer is None or current_time - self.kensatimer >= self.inspection_delay:
                self.kensatimer = current_time  # Update timer to current time

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                if self.cam_config.cowltop_last_inspect_maxredo == True:
                    rekensa_id = "yarinaoshi"
                else:
                    rekensa_id = "kensajisshi"

                dir_part = self.widget_dir_map.get(widgetidx)

                if dir_part:
                    base_dir = f"./aikensa/inspection_results/{dir_part}/nama"
                    before_img_path = f"{base_dir}/{timestamp}_{self.cam_config.kensainName}_{rekensa_id}_start.png"            
                    os.makedirs(base_dir, exist_ok=True)
                    cv2.imwrite(before_img_path, planarized)



                # if widgetidx == 5:
                #     before_img_path = f"./aikensa/inspection_results/66832A030P/nama/{timestamp}_{self.cam_config.kensainName}_{rekensa_id}_start.png"
                #     os.makedirs(
                #         "./aikensa/inspection_results/66832A030P/nama", exist_ok=True)
                #     cv2.imwrite(before_img_path, planarized)


                # Proceed with the inspection
                detections, det_frame = custom_infer_single(self.inferer, planarized, self.engine_config.conf_thres, self.engine_config.iou_thres, self.engine_config.max_det)
                # print(detections)
                
                if widgetidx == 5:
                    imgcheck, pitch_results, detected_pitch, total_length = partcheck_idx5(planarized, detections)

                if widgetidx == 6:
                    imgcheck, pitch_results, detected_pitch, total_length = partcheck_idx6(planarized, detections, partid="LH")

                if widgetidx == 7:
                    imgcheck, pitch_results, detected_pitch, total_length = partcheck_idx6(planarized, detections, partid="RH")

                detected_pitch = self.round_list_values(detected_pitch)  # Round the detected pitch values
                # Round the total length value
                total_length = self.round_values(total_length)


                if widgetidx == 5:
                        
                    if len(pitch_results) == len(self.cam_config.cowltoppitch):
                        self.cam_config.cowltoppitch = pitch_results

                    if all(result == 1 for result in pitch_results):
                        ok_count += 1  # All values are 1, increment OK count
                        self.cam_config.cowltop_last_inspection_outcome = True

                    else:
                        ng_count += 1  # At least one value is 0, increment NG coun
                        self.cam_config.cowltop_last_inspection_outcome = False

                    self.cam_config.cowltop_numofPart = (ok_count, ng_count)

                if widgetidx == 6:
                    if len(pitch_results) == len(self.cam_config.rrsideLHpitch):
                        self.cam_config.rrsideLHpitch = pitch_results

                    if all(result == 1 for result in pitch_results):
                        ok_count += 1

                    else:
                        ng_count += 1

                    self.cam_config.rrsideLHnumofPart = (ok_count, ng_count)

                if widgetidx == 7:
                    if len(pitch_results) == len(self.cam_config.rrsideRHpitch):
                        self.cam_config.rrsideRHpitch = pitch_results

                    if all(result == 1 for result in pitch_results):
                        ok_count += 1

                    else:
                        ng_count += 1

                    self.cam_config.rrsideRHnumofPart = (ok_count, ng_count)


                imgresults = imgcheck.copy()
                

                if widgetidx == 5:    
                    # Add the word "bundle now" into the image results if parts is divisible by 50
                    if ok_count % 50 == 0 and all(result == 1 for result in pitch_results):
                        imgresults = cv2.cvtColor(imgresults, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(imgresults)
                        font = ImageFont.truetype(self.kanjiFontPath, 60)
                        draw = ImageDraw.Draw(img_pil)
                        centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                        draw.text((centerpos[0]-250, centerpos[1]+180), u"束ねてください。", 
                                  font=font, fill=(50, 150, 150, 0))
                        imgresults = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                if widgetidx == 6 or widgetidx == 7:    
                    # Add the word "bundle now" into the image results if parts is divisible by 50
                    if ok_count % 10 == 0 and all(result == 1 for result in pitch_results):
                        imgresults = cv2.cvtColor(imgresults, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(imgresults)
                        font = ImageFont.truetype(self.kanjiFontPath, 60)
                        draw = ImageDraw.Draw(img_pil)
                        centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                        draw.text((centerpos[0]-250, centerpos[1]+180), u"束ねてください。", 
                                  font=font, fill=(50, 150, 150, 0))
                        imgresults = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        

                if dir_part:
                    base_dir = f"./aikensa/inspection_results/{dir_part}/kekka"
                    after_img_path = f"{base_dir}/{timestamp}_{self.cam_config.kensainName}_{rekensa_id}_zfinish.png"
                    os.makedirs(base_dir, exist_ok=True)
                    cv2.imwrite(after_img_path, imgresults)

                    if widgetidx == 5:
                        base_dir = f"./aikensa/inspection_results/{dir_part}/results"
                        os.makedirs(base_dir, exist_ok=True)

                        if not os.path.exists(f"{base_dir}/inspection_results.csv"):
                            with open(f"{base_dir}/inspection_results.csv", mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(['KensaResult(OK,/NG)', 'KensaTime', 'KensaSagyoushaName',
                                                'DetectedPitch', 'TotalLength', 'KensaYarinaoshi'])

                                writer.writerow([self.cam_config.cowltop_numofPart, timestamp,
                                                self.cam_config.kensainName, detected_pitch,
                                                total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                        else:
                            with open(f"{base_dir}/inspection_results.csv", mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([self.cam_config.cowltop_numofPart, timestamp,
                                                self.cam_config.kensainName, detected_pitch,
                                                total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                    if widgetidx == 6:
                        base_dir = f"./aikensa/inspection_results/{dir_part}/results"
                        os.makedirs(base_dir, exist_ok=True)

                        if not os.path.exists(f"{base_dir}/inspection_results.csv"):
                            with open(f"{base_dir}/inspection_results.csv", mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(['KensaResult(OK,/NG)', 'KensaTime', 'KensaSagyoushaName',
                                                'DetectedPitch', 'TotalLength', 'KensaYarinaoshi'])

                                writer.writerow([self.cam_config.rrsideLHnumofPart, timestamp,
                                                self.cam_config.kensainName, detected_pitch,
                                                total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                        else:
                            with open(f"{base_dir}/inspection_results.csv", mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([self.cam_config.rrsideLHnumofPart, timestamp,
                                                self.cam_config.kensainName, detected_pitch,
                                                total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                    if widgetidx == 7:
                        base_dir = f"./aikensa/inspection_results/{dir_part}/results"
                        os.makedirs(base_dir, exist_ok=True)

                        if not os.path.exists(f"{base_dir}/inspection_results.csv"):
                            with open(f"{base_dir}/inspection_results.csv", mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(['KensaResult(OK,/NG)', 'KensaTime', 'KensaSagyoushaName',
                                                'DetectedPitch', 'TotalLength', 'KensaYarinaoshi'])

                                writer.writerow([self.cam_config.rrsideRHnumofPart, timestamp,
                                                self.cam_config.kensainName, detected_pitch,
                                                total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                        else:
                            with open(f"{base_dir}/inspection_results.csv", mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([self.cam_config.rrsideRHnumofPart, timestamp,
                                                self.cam_config.kensainName, detected_pitch,
                                                total_length, self.cam_config.cowltop_last_inspect_maxredo])



                self.cam_config.cowltop_doInspect = False  # Reset the inspect flag
                self.cam_config.cowltop_last_inspect_maxredo = False  # Reset the max redo flag
                self.last_img_results = imgresults

        if self.cam_config.cowltop_doReinspect == True and self.cam_config.cowltop_last_inspect_maxredo == False:
            if self.kensatimer is None or current_time - self.kensatimer >= self.inspection_delay:
                self.kensatimer = current_time  # Update timer to current time

                # correct the inspection result based on last outcome
                if self.cam_config.cowltop_last_inspection_outcome is not None:

                    if self.cam_config.cowltop_last_inspection_outcome is True:
                        ok_count -= 1
                    elif self.cam_config.cowltop_last_inspection_outcome is False:
                        ng_count -= 1

                self.cam_config.cowltop_numofPart = (
                    ok_count, ng_count)
                self.cam_config.cowltop_doReinspect = False
                self.cam_config.cowltop_last_inspect_maxredo = True

        # Always check if we are within the inspection delay window for the inspection result
        if self.kensatimer and current_time - self.kensatimer < self.inspection_delay:
            qt_processed_frame = self.qt_processImage(
                self.last_img_results, width=1791, height=591)
        else:
            # Once the inspection delay has passed, revert back to the original planarized copy
            qt_processed_frame = self.qt_processImage(
                planarized_copy, width=1791, height=591)
            if self.kensatimer and current_time - self.kensatimer >= self.inspection_delay:
                self.cam_config.cowltoppitch = [0, 0, 0, 0, 0, 0]  # Reset pitch results
                self.cam_config.rrsideLHpitch = [0, 0, 0, 0, 0, 0, 0]  # Reset pitch results
                self.cam_config.rrsideRHpitch = [0, 0, 0, 0, 0, 0, 0]
                self.kensatimer = None  # Reset the timer

        # Emit the processed frame signal
        self.on_inference.emit(qt_processed_frame)

        self.cowl_pitch_updated.emit(self.cam_config.cowltoppitch)
        self.rrsideLH_pitch_updated.emit(self.cam_config.rrsideLHpitch)
        self.rrsideRH_pitch_updated.emit(self.cam_config.rrsideRHpitch)

        # print(f"Cowltoppitch: {self.cam_config.cowltoppitch}")
        # print(f"RRsideLHpitch: {self.cam_config.rrsideLHpitch}")
        # print(f"RRsideRHpitch: {self.cam_config.rrsideRHpitch}")

        self.cowl_numofPart_updated.emit(self.cam_config.cowltop_numofPart)
        self.rrsideLH_numofPart_updated.emit(self.cam_config.rrsideLHnumofPart)
        self.rrsideRH_numofPart_updated.emit(self.cam_config.rrsideRHnumofPart)



    def part_inspect_hood_rrsideLH(self, raw_frame):
        return None

    def part_inspect_hood_rrsideRH(self, raw_frame):
        return None

    def stop(self):
        self.running = False

    def qt_processImage(self, image, width=1791, height=731):
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = processed_image.shape
        bytesPerLine = ch * w
        processed_image = QImage(processed_image.data,
                                 w, h, bytesPerLine, QImage.Format_RGB888)
        processed_image = processed_image.scaled(
            width, height, Qt.KeepAspectRatio)

        return processed_image

    def rotate_frame(self, frame):
        # Rotate the frame 180 degrees
        return cv2.rotate(frame, cv2.ROTATE_180)

    def round_list_values(self, lst):
        return [round(x, 2) for x in lst]

    def round_values(self, value):
        return round(value, 2)

    def loadCalibrationParams(calibration_file_path):
        with open(calibration_file_path, 'r') as file:
            calibration_data = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_data.get('camera_matrix'))
            distortion_coefficients = np.array(
                calibration_data.get('distortion_coefficients'))
        return camera_matrix, distortion_coefficients

    def initialize_model(self, engine_config: EngineConfig = None):
        #Change based on the widget
        if self.cam_config.widget == 5:
            engine_config = EngineConfig(
                webcam=False,
                webcam_addr='0',
                img_size=1920,
                weights='./aikensa/custom_weights/cowltop_66832A030P.pt',
                device=0,
                yaml='./aikensa/custom_data/cowltop_66832A030P.yaml',
                conf_thres=0.4,
                iou_thres=0.45,
                max_det=1000
            )
        if self.cam_config.widget == 6 or self.cam_config.widget == 7:
            engine_config = EngineConfig(
                webcam=False,
                webcam_addr='0',
                img_size=1920,
                weights='./aikensa/custom_weights/hoodrrside_5902A5xx.pt',
                device=0,
                yaml='./aikensa/custom_data/hoodrrside_5902A5xx.yaml',
                conf_thres=0.4,
                iou_thres=0.7,
                max_det=1000
            )
        
        self.engine_config = engine_config

        # print (self.engine_config)
        self.inferer = create_inferer(engine_config)
