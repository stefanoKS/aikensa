import cv2
import os
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from camscripts.cam_init import initialize_camera
from PyQt5.QtGui import QImage, QPixmap
from opencv_imgprocessing.cannydetect import canny_edge_detection


class CameraThread(QThread):
    on_frame_raw = pyqtSignal(QImage)
    on_frame_processed = pyqtSignal(QImage)

    def __init__(self, config = None, capture=False):
        super(CameraThread, self).__init__()
        self.running = True


        if config is None:
            self.config = {
                "opacity": 0.5,
                "blur": 10,
                "lower_canny": 100,
                "upper_canny": 200,
                "contrast": 150,
                "brightness": 0,
                "capture": False
            }
        else:
            self.config = config

    def run(self):
        cap = initialize_camera()
        while self.running:
            ret, raw_frame = cap.read()
            if ret:
                processed_frame = canny_edge_detection(raw_frame, self.config["opacity"], self.config["blur"], self.config["lower_canny"], self.config["upper_canny"], self.config["contrast"], self.config["brightness"])
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = processed_frame.shape
                bytesPerLine = ch * w

                capture_frame = raw_frame.copy()
                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                
                processed_frame = QImage(processed_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                processed_frame = processed_frame.scaled(1791, 731, Qt.KeepAspectRatio)  
                raw_frame = QImage(raw_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                raw_frame = raw_frame.scaled(1791, 731, Qt.KeepAspectRatio)  
                
                if self.config["capture"] == "True":
                    
                    if not os.path.exists("./training_image"):
                        os.makedirs("./training_image")

                    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
                    file_name = f"capture_{current_time}.png"
                    cv2.imwrite(os.path.join("./training_image", file_name), capture_frame)    
                    self.config["capture"] = "False"
                
                self.on_frame_processed.emit(processed_frame)
                self.on_frame_raw.emit(raw_frame)

        cap.release()

    def stop(self):
        self.running = False
