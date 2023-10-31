from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from camscripts.cam_init import initialize_camera

from opencv_imgprocessing.cannydetect import canny_edge_detection
from opencv_imgprocessing.detectaruco import detectAruco
from opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix

import numpy as np
import cv2
import sys
import yaml
import os


# Camera Thread
class CameraThread(QThread):
    changePixmap = pyqtSignal(np.ndarray)

    def __init__(self):
        super(CameraThread, self).__init__()
        self.running = True


    def run(self):
        cap = initialize_camera()
        while self.running:
            ret, frame = cap.read()
            if ret:
                current_widget_index = stackedWidget.currentIndex()

                # Access the current widget and find slider, buttons, etc.
                current_widget = stackedWidget.widget(current_widget_index)

                if current_widget_index == 1:  # cameracalib.ui
                    takeimagebutton = current_widget.findChild(QPushButton, "takeimagebutton")
                    calculatebutton = current_widget.findChild(QPushButton, "calculatebutton")

                    if takeimagebutton:
                        detectCharucoBoard(frame)
                    if calculatebutton:
                        calculatecameramatrix(frame.shape[:2])


                if current_widget_index == 2:  # edgedetection.ui
                    
                    saveparambutton = current_widget.findChild(QPushButton, "saveparambutton")





                   
                    frame = canny_edge_detection(frame, value_opacity, value_blur, value_lowercanny, value_uppercanny, value_contrast, value_brightness)
                    
                    #Save params to yaml file
                    if saveparambutton:
                        params = {
                            'opacity': value_opacity,
                            'blur': value_blur,
                            'lower_canny': value_lowercanny,
                            'upper_canny': value_uppercanny,
                            'contrast': value_contrast,
                            'brightness': value_brightness
                        }

                        if not os.path.exists("./params"):
                            os.makedirs("./params")

                        with open('./params/cannyparams.yaml', 'w') as outfile:
                            yaml.dump(params, outfile, default_flow_style=False)


                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1791, 731, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
        cap.release()

    def stop(self):
        self.running = False


def close_app():
    cam_thread.stop()
    app.quit()

def load_ui(filename):
    widget = QMainWindow()
    loadUi(filename, widget)
    return widget

def set_frame(image):
    edge_detect_widget = stackedWidget.widget(2)  # Assuming 'edgedetection.ui' is at index 2
    label = edge_detect_widget.findChild(QLabel, "cameraFrame")
    label.setPixmap(QPixmap.fromImage(image))

def read_edgedetectslide():
        values = {
            "opacity": value_opacity = slider_opacity.value()/100,
            "blur": value_blur = slider_blur.value(),
            "lowercanny": value_lowercanny = slider_lowercanny.value(),
            "uppercanny": value_uppercanny = slider_uppercanny.value(),
            "contrast": value_contrast = slider_contrast.value()/100,
            "brightness": value_brightness = slider_brightness.value()/100,
        }

    update_slider_values.emit(values)

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    
    stackedWidget = QStackedWidget()
    
    # List of UI files to be loaded
    ui_files = ["./qtui/mainPage.ui", "./qtui/cameracalib.ui", "./qtui/edgedetection.ui", "./qtui/generatetrainingimage.ui", "./qtui/checkaruco.ui", "./qtui/66832A030P.ui"]
    
    for ui in ui_files:
        widget = load_ui(ui)
        stackedWidget.addWidget(widget)

    MainWindow.setCentralWidget(stackedWidget)
    stackedWidget.setCurrentIndex(0)
    
    # Camera thread setup
    cam_thread = CameraThread()
    cam_thread.changePixmap.connect(set_frame)
    cam_thread.start()

    # Find and connect buttons in main widget
    main_widget = stackedWidget.widget(0)
    calib_button = main_widget.findChild(QPushButton, "calibrationbutton")
    edgedetect_button = main_widget.findChild(QPushButton, "edgedetectbutton")
    generateimage_button = main_widget.findChild(QPushButton, "generateimagebutton")
    checkaruco_button = main_widget.findChild(QPushButton, "checkarucobutton")
    P66832A030P_button = main_widget.findChild(QPushButton, "P66832A030Pbutton")

    # Extra buttons on edgedetection.ui
    saveparambutton = edgedetect_button.findChild(QPushButton, "saveparambutton")
    # Sliders on edgedetection.ui
    slider_opacity = edgedetect_button.findChild(QSlider, "slider_opacity")
    slider_blur = edgedetect_button.findChild(QSlider, "slider_blur")
    slider_lowercanny = edgedetect_button.findChild(QSlider, "slider_lowercanny")
    slider_uppercanny = edgedetect_button.findChild(QSlider, "slider_uppercanny")
    slider_contrast = edgedetect_button.findChild(QSlider, "slider_contrast")
    slider_brightness = edgedetect_button.findChild(QSlider, "slider_brightness")



    if calib_button:
        calib_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(1))
    if edgedetect_button:
        edgedetect_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(2))
    if edgedetect_button:
        generateimage_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(3))
    if checkaruco_button:
        checkaruco_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(4))
    if P66832A030P_button:
        P66832A030P_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(5))


    

    # Find and connect quit buttons and main menu buttons in all widgets
    for i in range(stackedWidget.count()):
        widget = stackedWidget.widget(i)
        quit_button = widget.findChild(QPushButton, "quitbutton")
        main_menu_button = widget.findChild(QPushButton, "mainmenubutton")
        
        if quit_button:
            quit_button.clicked.connect(close_app)
        
        if main_menu_button:
            main_menu_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(0))
    
    MainWindow.showFullScreen()
    sys.exit(app.exec_())
