import cv2
import sys
import yaml
import os
from enum import Enum

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider, QMainWindow, QWidget, QCheckBox, QShortcut
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from aikensa.opencv_imgprocessing.cannydetect import canny_edge_detection
from aikensa.opencv_imgprocessing.detectaruco import detectAruco
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.cam_thread import CameraThread, CameraConfig

class WidgetUI(Enum):
    MAIN_PAGE = 0
    CAMERA_CALIB = 1
    EDGE_DETECTION = 2
    GENERATE_TRAINING_IMAGE = 3
    CHECK_ARUCO = 4
    P66832A030P = 5

# List of UI files to be loaded
UI_FILES = [
    'aikensa/qtui/mainPage.ui', #index 0
    'aikensa/qtui/cameracalib.ui', #index 1
    'aikensa/qtui/edgedetection.ui', #index 2
    'aikensa/qtui/generatetrainingimage.ui', #index 3
    'aikensa/qtui/checkaruco.ui', #index 4
    'aikensa/qtui/66832A030P.ui', #index 5
    
    # "./qtui/cameracalib.ui", #index 1
    # "./qtui/edgedetection.ui", #index 2
    # "./qtui/generatetrainingimage.ui", #index 3
    # "./qtui/checkaruco.ui", #index 4
    # "./qtui/66832A030P.ui", #index 5
] 


class AIKensa(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.cam_thread = CameraThread(CameraConfig())
        self._setup_ui()
        self.cam_thread.start()

    def _setup_ui(self):
        self.cam_thread.on_frame_raw.connect(self._set_frame_raw)
        self.cam_thread.on_frame_processed.connect(self._set_frame_processed)
        self.cam_thread.on_frame_aruco.connect(self._set_frame_aruco)
        self.cam_thread.on_inference.connect(self._set_frame_inference)
        self.cam_thread.cowl_pitch_updated.connect(self._set_button_color)
        
        self.stackedWidget = QStackedWidget()

        for ui in UI_FILES:
            widget = self._load_ui(ui)
            self.stackedWidget.addWidget(widget)

        self.stackedWidget.setCurrentIndex(0)
        
        main_widget = self.stackedWidget.widget(0)
        button_calib = main_widget.findChild(QPushButton, "calibrationbutton")
        button_edgedetect = main_widget.findChild(QPushButton, "edgedetectbutton")
        button_generateimage = main_widget.findChild(QPushButton, "generateimagebutton")
        button_checkaruco = main_widget.findChild(QPushButton, "checkarucobutton")
        button_P66832A030P = main_widget.findChild(QPushButton, "P66832A030Pbutton")
        
        if button_calib:
            button_calib.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
            button_calib.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 1))
        if button_edgedetect:
            button_edgedetect.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
            button_edgedetect.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 2))
        if button_generateimage:
            button_generateimage.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
            button_generateimage.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 3))
        if button_checkaruco:
            button_checkaruco.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(4))
            button_checkaruco.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 4))
            button_checkaruco.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'check_aruco', True))
        if button_P66832A030P:
            button_P66832A030P.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(5))
            button_P66832A030P.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 5))

        # Widget 1
        button_takeimagecalibrate = self.stackedWidget.widget(1).findChild(QPushButton, "takeimagebutton")
        button_takeimagecalibrate.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "capture", True))
        button_calculatecamparam = self.stackedWidget.widget(1).findChild(QPushButton, "calculatebutton")
        button_calculatecamparam.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "calculatecamparams", True))

        button_delcamcalibration = self.stackedWidget.widget(1).findChild(QPushButton, "delcalbbutton")
        button_delcamcalibration.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "delcamcalibration", True))

        # Widget 2
        button_saveparam = self.stackedWidget.widget(2).findChild(QPushButton, "saveparambutton")
        button_saveparam.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "savecannyparams", True))

        button_takecanny = self.stackedWidget.widget(2).findChild(QPushButton, "takeimagebutton")
        button_takecanny.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "capture", True))

        button_readwarp = self.stackedWidget.widget(2).findChild(QPushButton, "button_readwarp")
        label_readwarp = self.stackedWidget.widget(2).findChild(QLabel, "label_readwarpcolor")
        button_readwarp.pressed.connect(lambda: self._toggle_param_and_update_label("cannyreadwarp", label_readwarp))

        # Widget 3
        button_takeimage = self.stackedWidget.widget(3).findChild(QPushButton, "takeimagebutton")
        button_takeimage.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "capture", True))

        # Widget 4
        button_takearucoimage = self.stackedWidget.widget(4).findChild(QPushButton, "takeimagebutton")
        
        #frame = process_for_edge_detection(frame, self.slider_value)
        slider_opacity = self.stackedWidget.widget(2).findChild(QSlider, "slider_opacity")
        slider_blur = self.stackedWidget.widget(2).findChild(QSlider, "slider_blur")
        slider_lowercanny = self.stackedWidget.widget(2).findChild(QSlider, "slider_lowercanny")
        slider_uppercanny = self.stackedWidget.widget(2).findChild(QSlider, "slider_uppercanny")
        slider_contrast = self.stackedWidget.widget(2).findChild(QSlider, "slider_contrast")
        slider_brightness = self.stackedWidget.widget(2).findChild(QSlider, "slider_brightness")

        slider_opacity.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'opacity', x/100))
        slider_blur.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'blur', x))
        slider_lowercanny.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'lower_canny', x))
        slider_uppercanny.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'upper_canny', x))
        slider_contrast.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'contrast', x/100))
        slider_brightness.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'brightness', x/100))

        # Widget 5 -> CowlTop 66832A030P
        #_____________________________________________________________________________________________________
        #_____________________________________________________________________________________________________
        #_____________________________________________________________________________________________________
        #change value to true if false, false if true
        button_rtinference = self.stackedWidget.widget(5).findChild(QPushButton, "rtinferencebutton")
        label_rtinference = self.stackedWidget.widget(5).findChild(QLabel, "rtinferencecolor")
        button_rtinference.pressed.connect(lambda: self._toggle_param_and_update_label("rtinference", label_rtinference))

        button_rtwarp = self.stackedWidget.widget(5).findChild(QPushButton, "rtwarpbutton")
        label_rtwarp = self.stackedWidget.widget(5).findChild(QLabel, "rtwarpcolor")
        button_rtwarp.pressed.connect(lambda: self._toggle_param_and_update_label("rtwarp", label_rtwarp))

        button_savewarp = self.stackedWidget.widget(5).findChild(QPushButton, "savewarpbutton")
        button_savewarp.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "savewarp", True))
        #delwarpbutton
        button_delwarp = self.stackedWidget.widget(5).findChild(QPushButton, "delwarpbutton")
        button_delwarp.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "delwarp", True))




        button_kensa = self.stackedWidget.widget(5).findChild(QPushButton, "kensaButton")
        button_kensa.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "cowltop_doInspect", True))

        #add "b" button as shortcut for button_kensa
        self.shortcut_kensa = QShortcut(QKeySequence("b"), self)
        self.shortcut_kensa.activated.connect(button_kensa.click)
        


        #_____________________________________________________________________________________________________
        #_____________________________________________________________________________________________________
        #_____________________________________________________________________________________________________



        # Find and connect quit buttons and main menu buttons in all widgets
        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            button_quit = widget.findChild(QPushButton, "quitbutton")
            button_main_menu = widget.findChild(QPushButton, "mainmenubutton")
            
            if button_quit:
                button_quit.clicked.connect(self._close_app)
            
            if button_main_menu:
                button_main_menu.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                button_main_menu.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 0))
                #checkaruco.clicked.connect(lambda: set_params(self.cam_thread, 'check_aruco', False))
        
        
        self.setCentralWidget(self.stackedWidget)
        self.showFullScreen()
        

    def _close_app(self):
        self.cam_thread.stop()
        self.quit()

    def _load_ui(self, filename):
        widget = QMainWindow()
        loadUi(filename, widget)
        return widget

    def _set_frame_raw(self, image):
        for i in [1, 3]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "cameraFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _set_frame_processed(self, image):
        widget = self.stackedWidget.widget(2)
        label = widget.findChild(QLabel, "cameraFrame")
        label.setPixmap(QPixmap.fromImage(image))

    def _set_frame_aruco(self, image):
        widget = self.stackedWidget.widget(4)
        label = widget.findChild(QLabel, "cameraFrame")
        label.setPixmap(QPixmap.fromImage(image))

    def _set_frame_inference(self, image):
        widget = self.stackedWidget.widget(5)
        label = widget.findChild(QLabel, "cameraFrame")
        label.setPixmap(QPixmap.fromImage(image))

    def _set_cam_params(self, thread, key, value):
        setattr(thread.cam_config, key, value)

    def _toggle_param_and_update_label(self, param, label):
        # Toggle the parameter value
        new_value = not getattr(self.cam_thread.cam_config, param)
        self._set_cam_params(self.cam_thread, param, new_value)

        # Update the label color based on the new parameter value
        color = "green" if new_value else "red"
        label.setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _cowltop_update_label(self, param, pitchvalue,labels):
        pitch = getattr(self.cam_thread.cam_config, pitchvalue)
        self._set_cam_params(self.cam_thread, param, True)
        colorOK = "green" 
        colorNG = "red"
        #if the pitchvalue[i] is 1, then labels[i] is green, else red
        for i in range(len(pitch)):
            color = colorOK if pitch[i] else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _set_button_color(self, pitch_data):
        colorOK = "green" 
        colorNG = "red"

        label_names = ["P1color", "P2color", "P3color", "P4color", "P5color", "Lsuncolor"]
        labels = [self.stackedWidget.widget(5).findChild(QLabel, name) for name in label_names]
        for i, pitch_value in enumerate(pitch_data):
            color = colorOK if pitch_value else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")




def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()