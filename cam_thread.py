import cv2

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from camscripts.cam_init import initialize_camera
from PyQt5.QtGui import QImage, QPixmap
from opencv_imgprocessing.cannydetect import canny_edge_detection


class CameraThread(QThread):
    on_frame = pyqtSignal(QImage)

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
            ret, frame = cap.read()
            if ret:
                # print(self.config)
                frame = canny_edge_detection(frame, self.config["opacity"], self.config["blur"], self.config["lower_canny"], self.config["upper_canny"], self.config["contrast"], self.config["brightness"])
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1791, 731, Qt.KeepAspectRatio)  
                self.on_frame.emit(p)

        cap.release()

    def stop(self):
        self.running = False
