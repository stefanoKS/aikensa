from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap


from opencv_imgprocessing.cannydetect import canny_edge_detection
from opencv_imgprocessing.detectaruco import detectAruco
from opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix

from cam_thread import CameraThread

import cv2
import sys
import yaml
import os


def close_app():
    cam_thread.stop()
    app.quit()

def load_ui(filename):
    widget = QMainWindow()
    loadUi(filename, widget)
    return widget

def set_frame_raw(image):
    for i in [1, 3, 4]:
        widget = stackedWidget.widget(i)
        label = widget.findChild(QLabel, "cameraFrame")
        label.setPixmap(QPixmap.fromImage(image))

def set_frame_processed(image):
    widget = stackedWidget.widget(2)
    label = widget.findChild(QLabel, "cameraFrame")
    label.setPixmap(QPixmap.fromImage(image))


def set_params(thread, key, value):
    thread.config[key] = value


if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()

    # Camera thread setup
    cam_thread = CameraThread()
    cam_thread.on_frame_raw.connect(set_frame_raw)
    cam_thread.on_frame_processed.connect(set_frame_processed)
    
    stackedWidget = QStackedWidget()
    
    # List of UI files to be loaded
    ui_files = ["./qtui/mainPage.ui", "./qtui/cameracalib.ui", "./qtui/edgedetection.ui", "./qtui/generatetrainingimage.ui", "./qtui/checkaruco.ui", "./qtui/66832A030P.ui"]
    
    for ui in ui_files:
        widget = load_ui(ui)
        stackedWidget.addWidget(widget)

    MainWindow.setCentralWidget(stackedWidget)
    stackedWidget.setCurrentIndex(0)
    

    

    # Find and connect buttons in main widget
    main_widget = stackedWidget.widget(0)
    calib_button = main_widget.findChild(QPushButton, "calibrationbutton")
    edgedetect_button = main_widget.findChild(QPushButton, "edgedetectbutton")
    generateimage_button = main_widget.findChild(QPushButton, "generateimagebutton")
    checkaruco_button = main_widget.findChild(QPushButton, "checkarucobutton")
    P66832A030P_button = main_widget.findChild(QPushButton, "P66832A030Pbutton")

    

    if calib_button:
        calib_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(1))
    if edgedetect_button:
        edgedetect_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(2))
    if generateimage_button:
        generateimage_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(3))
    if checkaruco_button:
        checkaruco_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(4))
    if P66832A030P_button:
        P66832A030P_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(5))


    # current_widget_index = stackedWidget.currentIndex()
    # current_widget = stackedWidget.widget(current_widget_index)

    # Widget 1
    takeimagebutton = stackedWidget.widget(1).findChild(QPushButton, "takeimagebutton")
    calculatebutton = stackedWidget.widget(1).findChild(QPushButton, "calculatebutton")


    # Widget 2
    saveparambutton = stackedWidget.widget(2).findChild(QPushButton, "saveparambutton")

    # Widget 3
    takeimage_button = stackedWidget.widget(3).findChild(QPushButton, "takeimagebutton")
   
    takeimage_button.pressed.connect(lambda: set_params(cam_thread, "capture", "True"))

    

    #frame = process_for_edge_detection(frame, self.slider_value)
    slider_opacity = stackedWidget.widget(2).findChild(QSlider, "slider_opacity")
    slider_blur = stackedWidget.widget(2).findChild(QSlider, "slider_blur")
    slider_lowercanny = stackedWidget.widget(2).findChild(QSlider, "slider_lowercanny")
    slider_uppercanny = stackedWidget.widget(2).findChild(QSlider, "slider_uppercanny")
    slider_contrast = stackedWidget.widget(2).findChild(QSlider, "slider_contrast")
    slider_brightness = stackedWidget.widget(2).findChild(QSlider, "slider_brightness")

    slider_opacity.valueChanged.connect(lambda x: set_params(cam_thread, 'opacity', x/100))
    slider_blur.valueChanged.connect(lambda x: set_params(cam_thread, 'blur', x))
    slider_lowercanny.valueChanged.connect(lambda x: set_params(cam_thread, 'lower_canny', x))
    slider_uppercanny.valueChanged.connect(lambda x: set_params(cam_thread, 'upper_canny', x))
    slider_contrast.valueChanged.connect(lambda x: set_params(cam_thread, 'contrast', x/100))
    slider_brightness.valueChanged.connect(lambda x: set_params(cam_thread, 'brightness', x/100))

    value_opacity = slider_opacity.value()/100
    value_blur = slider_blur.value()
    value_lowercanny = slider_lowercanny.value()
    value_uppercanny = slider_uppercanny.value()
    value_contrast = slider_contrast.value()/100
    value_brightness = slider_brightness.value()/100
    
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

    # Find and connect quit buttons and main menu buttons in all widgets
    for i in range(stackedWidget.count()):
        widget = stackedWidget.widget(i)
        quit_button = widget.findChild(QPushButton, "quitbutton")
        main_menu_button = widget.findChild(QPushButton, "mainmenubutton")
        
        if quit_button:
            quit_button.clicked.connect(close_app)
        
        if main_menu_button:
            main_menu_button.clicked.connect(lambda: stackedWidget.setCurrentIndex(0))
    

    
    cam_thread.start()

    MainWindow.showFullScreen()
    sys.exit(app.exec_())
