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

from aikensa.parts_config.daily_tenken_rrside import partcheck_pitch as partcheck_idx21
from aikensa.parts_config.daily_tenken_rrside import partcheck_color as partcheck_idx22
from aikensa.parts_config.daily_tenken_rrside import partcheck_hanire as partcheck_idx23
from aikensa.parts_config.daily_tenken_cowltop import partcheck as partcheck_idx24

#from aikensa.parts_config.hoodrrsideRH_5902A510 import partcheck as partcheck_idx7

from PIL import ImageFont, ImageDraw, Image

from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_keisoku_sound, play_konpou_sound


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
    #General Functions
    furyou_plus: bool = False
    furyou_minus: bool = False
    kansei_plus: bool = False
    kansei_minus: bool = False


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

    inspection_delay = 2.5

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

        self.inspection_result = False
        self.prev_timestamp = None

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

                if self.cam_config.widget == 5:
                    self.part_inspect(raw_frame, 5)

                if self.cam_config.widget == 6:
                    self.part_inspect(raw_frame, 6)

                if self.cam_config.widget == 7:
                    self.part_inspect(raw_frame, 7)

                # For Daily Tenken
                if self.cam_config.widget == 21:
                    self.part_inspect(raw_frame, 21)
                if self.cam_config.widget == 22:
                    self.part_inspect(raw_frame, 22)
                if self.cam_config.widget == 23:
                    self.part_inspect(raw_frame, 23)
                if self.cam_config.widget == 24:
                    self.part_inspect(raw_frame, 24)
                ##

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
        qt_processed_frame = self.qt_processImage(planarized_copy, width=1791, height=591)
        


        # for testing purpose, read image from directory as the qt_processed_frame
        image_path = "./aikensa/inspection_results/temp_image/test.png"  
        planarized = cv2.imread(image_path)  
        qt_processed_frame = planarized



        save_image_nama = planarized.copy()

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

            #manual adjustment for no of ok and 
            self.cam_config.cowltop_numofPart = self.manual_adjustment(ok_count, ng_count, self.cam_config.furyou_plus, self.cam_config.furyou_minus, self.cam_config.kansei_plus, self.cam_config.kansei_minus)


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

            self.cam_config.rrsideLHnumofPart = self.manual_adjustment(ok_count, ng_count, self.cam_config.furyou_plus, self.cam_config.furyou_minus, self.cam_config.kansei_plus, self.cam_config.kansei_minus)

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

            self.cam_config.rrsideRHnumofPart = self.manual_adjustment(ok_count, ng_count, self.cam_config.furyou_plus, self.cam_config.furyou_minus, self.cam_config.kansei_plus, self.cam_config.kansei_minus)


        # Check if the inspection flag is True
        if self.cam_config.cowltop_doInspect == True:
            if self.kensatimer is None or current_time - self.kensatimer >= self.inspection_delay:
                self.kensatimer = current_time  # Update timer to current time


                if self.prev_timestamp == None:
                    self.prev_timestamp = datetime.now()

                timestamp = datetime.now() #datetime.now().strftime('%Y%m%d_%H%M%S')

                deltaTime = timestamp - self.prev_timestamp
                self.prev_timestamp = timestamp

                if self.cam_config.cowltop_last_inspect_maxredo == True:
                    rekensa_id = "yarinaoshi"
                else:
                    rekensa_id = "kensajisshi"

                dir_part = self.widget_dir_map.get(widgetidx)

                detected_pitch = []
                total_length = 0

                # if dir_part:
                #     base_dir = f"./aikensa/inspection_results/{dir_part}/nama"
                #     before_img_path = f"{base_dir}/{timestamp}_{self.cam_config.kensainName}_{rekensa_id}_start.png"            
                #     os.makedirs(base_dir, exist_ok=True)
                #     cv2.imwrite(before_img_path, planarized)

                if widgetidx == 5:
                    detections, _ = custom_infer_single(self.inferer_cowltop, planarized, 
                                                                self.engine_config_cowltop.conf_thres, 
                                                                self.engine_config_cowltop.iou_thres, 
                                                                self.engine_config_cowltop.max_det)
                    
                    imgcheck, pitch_results, detected_pitch, delta_pitch, total_length = partcheck_idx5(planarized, detections)

                if widgetidx == 6:
                    detections, _ = custom_infer_single(self.inferer_rrside, planarized,
                                                                self.engine_config_rrside.conf_thres, 
                                                                self.engine_config_rrside.iou_thres, 
                                                                self.engine_config_rrside.max_det)
                    
                    detections_hanire, _ = custom_infer_single(self.inferer_rrside_hanire, planarized,
                                                                self.engine_config_custom_hanire.conf_thres, 
                                                                self.engine_config_custom_hanire.iou_thres, 
                                                                self.engine_config_custom_hanire.max_det)
                    
                    imgcheck, pitch_results, detected_pitch, delta_pitch, total_length, hanire = partcheck_idx6(planarized, detections, detections_hanire, partid="LH")

                if widgetidx == 7:
                    detections, _ = custom_infer_single(self.inferer_rrside, planarized,
                                                                self.engine_config_rrside.conf_thres, 
                                                                self.engine_config_rrside.iou_thres, 
                                                                self.engine_config_rrside.max_det)
                    
                    detections_hanire, _ = custom_infer_single(self.inferer_rrside_hanire, planarized,
                                                                self.engine_config_custom_hanire.conf_thres, 
                                                                self.engine_config_custom_hanire.iou_thres, 
                                                                self.engine_config_custom_hanire.max_det)
                    
                    imgcheck, pitch_results, detected_pitch, delta_pitch, total_length , hanire= partcheck_idx6(planarized, detections, detections_hanire, partid="RH")

                if widgetidx == 21:
                    detections, _ = custom_infer_single(self.inferer_rrside, planarized,
                                                                self.engine_config_rrside.conf_thres, 
                                                                self.engine_config_rrside.iou_thres, 
                                                                self.engine_config_rrside.max_det)
                    
                    imgcheck, pitch_results, detected_pitch, total_length = partcheck_idx21(planarized, detections)
                
                if widgetidx == 22:
                    detections, _ = custom_infer_single(self.inferer_rrside, planarized,
                                                                self.engine_config_rrside.conf_thres, 
                                                                self.engine_config_rrside.iou_thres, 
                                                                self.engine_config_rrside.max_det)
                    
                    imgcheck, pitch_results = partcheck_idx22(planarized, detections)

                if widgetidx == 23:
                    detections, _ = custom_infer_single(self.inferer_rrside_hanire, planarized,
                                                                self.engine_config_custom_hanire.conf_thres, 
                                                                self.engine_config_custom_hanire.iou_thres, 
                                                                self.engine_config_custom_hanire.max_det)

                    imgcheck, pitch_results = partcheck_idx23(planarized, detections)

                if widgetidx == 24:
                    detections, _ = custom_infer_single(self.inferer_cowltop, planarized, 
                                                                self.engine_config_cowltop.conf_thres, 
                                                                self.engine_config_cowltop.iou_thres, 
                                                                self.engine_config_cowltop.max_det)
                    
                    imgcheck, pitch_results, detected_pitch, total_length = partcheck_idx24(planarized, detections)

                detected_pitch = self.round_list_values(detected_pitch)  # Round the detected pitch values
                delta_pitch = self.round_list_values(delta_pitch)  # Round the delta pitch value
                # Round the total length value
                total_length = self.round_values(total_length)


                if widgetidx == 5:
                        
                    if len(pitch_results) == len(self.cam_config.cowltoppitch):
                        self.cam_config.cowltoppitch = pitch_results

                    if all(result == 1 for result in pitch_results):
                        ok_count += 1  # All values are 1, increment OK count
                        self.cam_config.cowltop_last_inspection_outcome = True
                        self.inspection_result = True

                    else:
                        ng_count += 1  # At least one value is 0, increment NG coun
                        self.cam_config.cowltop_last_inspection_outcome = False
                        self.inspection_result = False

                    self.cam_config.cowltop_numofPart = (ok_count, ng_count)

                if widgetidx == 6:
                    if len(pitch_results) == len(self.cam_config.rrsideLHpitch):
                        self.cam_config.rrsideLHpitch = pitch_results

                    if all(result == 1 for result in pitch_results) and not hanire:
                        ok_count += 1
                        self.inspection_result = True

                    else:
                        ng_count += 1
                        self.inspection_result = False

                    self.cam_config.rrsideLHnumofPart = (ok_count, ng_count)

                if widgetidx == 7:
                    if len(pitch_results) == len(self.cam_config.rrsideRHpitch):
                        self.cam_config.rrsideRHpitch = pitch_results
                        
                    if all(result == 1 for result in pitch_results) and not hanire:
                        ok_count += 1
                        self.inspection_result = True

                    else:
                        ng_count += 1
                        self.inspection_result = False

                    self.cam_config.rrsideRHnumofPart = (ok_count, ng_count)


                imgresults = imgcheck.copy()
                

                if widgetidx == 5:    
                    # Add the word "bundle now" into the image results if parts is divisible by 50
                    if ok_count % 50 == 0 and all(result == 1 for result in pitch_results):
                        play_keisoku_sound()
                        imgresults = cv2.cvtColor(imgresults, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(imgresults)
                        font = ImageFont.truetype(self.kanjiFontPath, 60)
                        draw = ImageDraw.Draw(img_pil)
                        centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                        draw.text((centerpos[0]-500, centerpos[1]+180), u"束ねてください。", 
                                  font=font, fill=(5, 30, 50, 0))
                        imgresults = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                if widgetidx == 6 or widgetidx == 7:    
                    # Add the word "bundle now" into the image results if parts is divisible by 50
                    if ok_count % 10 == 0 and all(result == 1 for result in pitch_results):
                        if ok_count % 150 == 0:
                            imgresults = cv2.cvtColor(imgresults, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(imgresults)
                            font = ImageFont.truetype(self.kanjiFontPath, 60)
                            draw = ImageDraw.Draw(img_pil)
                            centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            draw.text((centerpos[0]-500, centerpos[1]+180), u"ダンボールに入れてください", 
                                    font=font, fill=(5, 50, 210, 0))
                            imgresults = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            play_konpou_sound()
                            
                        else:
                            imgresults = cv2.cvtColor(imgresults, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(imgresults)
                            font = ImageFont.truetype(self.kanjiFontPath, 60)
                            draw = ImageDraw.Draw(img_pil)
                            centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            draw.text((centerpos[0]-500, centerpos[1]+180), u"束ねてください。", 
                                      font=font, fill=(5, 30, 50, 0))
                            imgresults = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            play_keisoku_sound()
                            
        

                # if dir_part:
                #     base_dir = f"./aikensa/inspection_results/{dir_part}/kekka"
                #     after_img_path = f"{base_dir}/{timestamp}_{self.cam_config.kensainName}_{rekensa_id}_zfinish.png"
                #     os.makedirs(base_dir, exist_ok=True)
                #     cv2.imwrite(after_img_path, imgresults)

                #     if widgetidx == 5:
                #         base_dir = f"./aikensa/inspection_results/{dir_part}/results"
                #         os.makedirs(base_dir, exist_ok=True)

                #         if not os.path.exists(f"{base_dir}/inspection_results.csv"):
                #             with open(f"{base_dir}/inspection_results.csv", mode='w', newline='') as file:
                #                 writer = csv.writer(file)
                #                 writer.writerow(['KensaResult(OK,/NG)', 'KensaTime', 'KensaSagyoushaName',
                #                                 'DetectedPitch', 'TotalLength', 'KensaYarinaoshi'])

                #                 writer.writerow([self.cam_config.cowltop_numofPart, timestamp,
                #                                 self.cam_config.kensainName, detected_pitch,
                #                                 total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                #         else:
                #             with open(f"{base_dir}/inspection_results.csv", mode='a', newline='') as file:
                #                 writer = csv.writer(file)
                #                 writer.writerow([self.cam_config.cowltop_numofPart, timestamp,
                #                                 self.cam_config.kensainName, detected_pitch,
                #                                 total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                #     if widgetidx == 6:
                #         base_dir = f"./aikensa/inspection_results/{dir_part}/results"
                #         os.makedirs(base_dir, exist_ok=True)

                #         if not os.path.exists(f"{base_dir}/inspection_results.csv"):
                #             with open(f"{base_dir}/inspection_results.csv", mode='w', newline='') as file:
                #                 writer = csv.writer(file)
                #                 writer.writerow(['KensaResult(OK,/NG)', 'KensaTime', 'KensaSagyoushaName',
                #                                 'DetectedPitch', 'TotalLength', 'KensaYarinaoshi'])

                #                 writer.writerow([self.cam_config.rrsideLHnumofPart, timestamp,
                #                                 self.cam_config.kensainName, detected_pitch,
                #                                 total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                #         else:
                #             with open(f"{base_dir}/inspection_results.csv", mode='a', newline='') as file:
                #                 writer = csv.writer(file)
                #                 writer.writerow([self.cam_config.rrsideLHnumofPart, timestamp,
                #                                 self.cam_config.kensainName, detected_pitch,
                #                                 total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                #     if widgetidx == 7:
                #         base_dir = f"./aikensa/inspection_results/{dir_part}/results"
                #         os.makedirs(base_dir, exist_ok=True)

                #         if not os.path.exists(f"{base_dir}/inspection_results.csv"):
                #             with open(f"{base_dir}/inspection_results.csv", mode='w', newline='') as file:
                #                 writer = csv.writer(file)
                #                 writer.writerow(['KensaResult(OK,/NG)', 'KensaTime', 'KensaSagyoushaName',
                #                                 'DetectedPitch', 'TotalLength', 'KensaYarinaoshi'])

                #                 writer.writerow([self.cam_config.rrsideRHnumofPart, timestamp,
                #                                 self.cam_config.kensainName, detected_pitch,
                #                                 total_length, self.cam_config.cowltop_last_inspect_maxredo])
                                
                #         else:
                #             with open(f"{base_dir}/inspection_results.csv", mode='a', newline='') as file:
                #                 writer = csv.writer(file)
                #                 writer.writerow([self.cam_config.rrsideRHnumofPart, timestamp,
                #                                 self.cam_config.kensainName, detected_pitch,
                #                                 total_length, self.cam_config.cowltop_last_inspect_maxredo])

                save_image_kekka = imgresults

                #save image to resepected directory
                self.save_image(dir_part, save_image_nama, save_image_kekka, timestamp, self.cam_config.kensainName, self.inspection_result, rekensa_id)
                if widgetidx == 5:
                    self.save_result_csv(dir_part, self.cam_config.cowltop_numofPart, timestamp, deltaTime, self.cam_config.kensainName, detected_pitch, delta_pitch, total_length, self.cam_config.cowltop_last_inspect_maxredo)
                if widgetidx == 6:
                    self.save_result_csv(dir_part, self.cam_config.rrsideLHnumofPart, timestamp, deltaTime, self.cam_config.kensainName, detected_pitch, delta_pitch, total_length, self.cam_config.cowltop_last_inspect_maxredo)
                if widgetidx == 7:
                    self.save_result_csv(dir_part , self.cam_config.rrsideRHnumofPart, timestamp, deltaTime, self.cam_config.kensainName, detected_pitch, delta_pitch, total_length, self.cam_config.cowltop_last_inspect_maxredo)

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
            qt_processed_frame = self.qt_processImage(planarized_copy, width=1791, height=591)
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

    def save_image(self, dir_part, save_image_nama, save_image_kekka, timestamp, kensainName, inspection_result, rekensa_id):
        if inspection_result == True:
            resultid = "OK"
        else:
            resultid = "NG"

        timestamp_date = timestamp.strftime("%Y%m%d")
        timestamp_hour = timestamp.strftime("%H%M%S")


        base_dir_nama = f"./aikensa/inspection_results/{dir_part}/{timestamp_date}/{resultid}/nama"
        base_dir_kekka = f"./aikensa/inspection_results/{dir_part}/{timestamp_date}/{resultid}/kekka"

        img_path_nama = f"{base_dir_nama}/{timestamp_hour}_{kensainName}_start.png"
        img_path_kekka = f"{base_dir_kekka}/{timestamp_hour}_{kensainName}_finish.png"

        os.makedirs(base_dir_nama, exist_ok=True)
        os.makedirs(base_dir_kekka, exist_ok=True)

        cv2.imwrite(img_path_nama, save_image_nama)
        cv2.imwrite(img_path_kekka, save_image_kekka)

    def save_result_csv(self, dir_part, numofPart, timestamp, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, cowltop_last_inspect_maxredo):
        detected_pitch_str = str(detected_pitch).replace('[', '').replace(']', '')
        delta_pitch_str = str(delta_pitch).replace('[', '').replace(']', '')

        timestamp_date = timestamp.strftime("%Y%m%d")
        timestamp_hour = timestamp.strftime("%H%M%S")
        deltaTime = deltaTime.total_seconds()

        base_dir = f"./aikensa/inspection_results/{dir_part}/{timestamp_date}/results"
        os.makedirs(base_dir, exist_ok=True)


        if not os.path.exists(f"{base_dir}/inspection_results.csv"):
            with open(f"{base_dir}/inspection_results.csv", mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['KensaResult(OK,/NG)', 'KensaTime', "KensaTimeLength", 'KensaSagyoushaName',
                                'DetectedPitch', "DeltaPitch", 'TotalLength', 'KensaYarinaoshi'])

                writer.writerow([numofPart, timestamp_hour, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, cowltop_last_inspect_maxredo])
                
        else:
            with open(f"{base_dir}/inspection_results.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([numofPart, timestamp_hour, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, cowltop_last_inspect_maxredo])
                
        # base_dir = f"./aikensa/inspection_results/{dir_part}/results"
        # os.makedirs(base_dir, exist_ok=True)

        # if not os.path.exists(f"{base_dir}/inspection_results.csv"):
        #     with open(f"{base_dir}/inspection_results.csv", mode='w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(['KensaResult(OK,/NG)', 'KensaTime', 'KensaSagyoushaName',
        #                         'DetectedPitch', 'TotalLength', 'KensaYarinaoshi'])

        #         writer.writerow([numofPart, timestamp, kensainName, detected_pitch, total_length, cowltop_last_inspect_maxredo])
                
        # else:
        #     with open(f"{base_dir}/inspection_results.csv", mode='a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([numofPart, timestamp, kensainName, detected_pitch, total_length, cowltop_last_inspect_maxredo])

    def manual_adjustment(self, ok_count, ng_count, furyou_plus, furyou_minus, kansei_plus, kansei_minus):
        if furyou_plus:
            ng_count += 1
            self.cam_config.furyou_plus = False

        if furyou_minus and ng_count > 0:
            ng_count -= 1
            self.cam_config.furyou_minus = False

        if kansei_plus:
            ok_count += 1
            self.cam_config.kansei_plus = False

        if kansei_minus and ok_count > 0:
            ok_count -= 1
            self.cam_config.kansei_minus = False

        return ok_count, ng_count
    
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
        engine_config_cowltop = None
        engine_config_rrside = None
        engine_config_rrside_hanire = None

        if self.cam_config.widget == 5 or self.cam_config.widget in [24]:
            engine_config_cowltop = EngineConfig(
                webcam=False,
                webcam_addr='0',
                img_size=1920,
                weights='./aikensa/custom_weights/cowltop_66832A030P.pt',
                device=0,
                yaml='./aikensa/custom_data/cowltop_66832A030P.yaml',
                conf_thres=0.6,
                iou_thres=0.45,
                max_det=1000
            )
            engine_config_custom = None
        if self.cam_config.widget in [6, 7] or self.cam_config.widget in [21, 22, 23]:
            engine_config_rrside = EngineConfig(
                webcam=False,
                webcam_addr='0',
                img_size=1920,
                weights='./aikensa/custom_weights/hoodrrside_5902A5xx.pt',
                device=0,
                yaml='./aikensa/custom_data/hoodrrside_5902A5xx.yaml',
                conf_thres=0.7,
                iou_thres=0.7,
                max_det=1000
            )
            engine_config_rrside_hanire = EngineConfig(
                webcam=False,
                webcam_addr='0',
                img_size=1920,
                weights='./aikensa/custom_weights/hoodrrside_hanire.pt',
                device=0,
                yaml='./aikensa/custom_data/hoodrrside_hanire.yaml',
                conf_thres=0.5,
                iou_thres=0.7,
                max_det=1000
            )

        if engine_config_cowltop:
            self.engine_config_cowltop = engine_config_cowltop
            self.inferer_cowltop = create_inferer(engine_config_cowltop)

        if engine_config_rrside:
            self.engine_config_rrside = engine_config_rrside
            self.inferer_rrside = create_inferer(engine_config_rrside)

        if engine_config_rrside_hanire:
            self.engine_config_custom_hanire = engine_config_rrside_hanire
            self.inferer_rrside_hanire = create_inferer(engine_config_rrside_hanire)
