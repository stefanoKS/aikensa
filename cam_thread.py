import cv2

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from camscripts.cam_init import initialize_camera
from PyQt5.QtGui import QImage, QPixmap
from opencv_imgprocessing.cannydetect import canny_edge_detection


class CameraThread(QThread):
    on_frame_raw = pyqtSignal(QImage)
    on_frame_processed = pyqtSignal(QImage)

    def __init__(self, config = None):
        super(CameraThread, self).__init__()
        self.running = True

        if config is None:
            self.config = {
                "opacity": 0.5,
                "blur": 10,
                "lower_canny": 100,
                "upper_canny": 200,
                "contrast": 150,
                "brightness": 100
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

                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                
                processed_frame = QImage(processed_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                processed_frame = processed_frame.scaled(1791, 731, Qt.KeepAspectRatio)  
                raw_frame = QImage(raw_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                raw_frame = raw_frame.scaled(1791, 731, Qt.KeepAspectRatio)  
                
                self.on_frame_processed.emit(processed_frame)
                self.on_frame_raw.emit(raw_frame)

        cap.release()

    def stop(self):
        self.running = False
