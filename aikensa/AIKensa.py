import cv2
import sys
import yaml
import os
from enum import Enum

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider, QMainWindow, QWidget, QCheckBox, QShortcut, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from aikensa.opencv_imgprocessing.cannydetect import canny_edge_detection
from aikensa.opencv_imgprocessing.detectaruco import detectAruco
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.cam_thread import CameraThread, CameraConfig

from aikensa.sio_thread import ServerMonitorThread


class WidgetUI(Enum):
    MAIN_PAGE = 0
    CAMERA_CALIB = 1
    EDGE_DETECTION = 2
    GENERATE_TRAINING_IMAGE = 3
    CHECK_ARUCO = 4
    P66832A030P = 5


# List of UI files to be loaded
UI_FILES = [
    'aikensa/qtui/mainPage.ui',  # index 0
    'aikensa/qtui/cameracalib.ui',  # index 1
    'aikensa/qtui/edgedetection.ui',  # index 2
    'aikensa/qtui/generatetrainingimage.ui',  # index 3
    'aikensa/qtui/checkaruco.ui',  # index 4
    'aikensa/qtui/66832A030P.ui',  # index 5
    "aikensa/qtui/5902A509.ui",  # index 6
    "aikensa/qtui/5902A510.ui",  # index 7
]


class AIKensa(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.cam_thread = CameraThread(CameraConfig())
        self._setup_ui()
        self.cam_thread.start()

        # Thread for SiO
        HOST = '192.168.0.100'  # Use the IP address from SiO settings
        PORT = 30001  # Use the port number from SiO settings

        self.server_monitor_thread = ServerMonitorThread(
            HOST, PORT, check_interval=0.1)
        self.server_monitor_thread.server_status_signal.connect(self.handle_server_status)
        self.server_monitor_thread.input_states_signal.connect(self.handle_input_states)
        self.server_monitor_thread.start()

    # def handle_server_status(self, is_up):
    #     if is_up:
    #         # print("Sio Server is up!")
    #         self.siostatus.setText("ON")
    #         self.siostatus.setStyleSheet("color: green;")
    #         self.siostatus_cowltop.setText("ON")
    #         self.siostatus_cowltop.setStyleSheet("color: green;")

    #     else:
    #         # print("Sio Server is down!")
    #         self.siostatus.setText("OFF")
    #         self.siostatus.setStyleSheet("color: red;")
    #         self.siostatus_cowltop.setText("OFF")
    #         self.siostatus_cowltop.setStyleSheet("color: red;")

    def handle_server_status(self, is_up):
        status_text = "ON" if is_up else "OFF"
        status_color = "green" if is_up else "red"

        for label in self.siostatus_cowltop:
            if label:  # Check if the label was found correctly
                label.setText(status_text)
                label.setStyleSheet(f"color: {status_color};")


    def handle_input_states(self, input_states):
        # check if input_stats is not empty
        if input_states:
            if input_states[0] == 1:
                self.trigger_kensa()
            if input_states[1] == 1:
                self.trigger_rekensa()
            else:
                pass

    def trigger_kensa(self):
        self.button_kensa5.click()
        self.button_kensa6.click()
        self.button_kensa7.click()

    def trigger_rekensa(self):
        self.button_rekensa.click()

    def _setup_ui(self):
        self.cam_thread.on_frame_raw.connect(self._set_frame_raw)
        self.cam_thread.on_frame_processed.connect(self._set_frame_processed)
        self.cam_thread.on_frame_aruco.connect(self._set_frame_aruco)
        self.cam_thread.on_inference.connect(self._set_frame_inference)
        
        self.cam_thread.cowl_pitch_updated.connect(self._set_button_color)
        self.cam_thread.cowl_numofPart_updated.connect(self._set_numlabel_text)

        self.cam_thread.rrsideLH_pitch_updated.connect(self._set_button_color_rrside_LH)
        self.cam_thread.rrsideLH_numofPart_updated.connect(self._set_numlabel_text_rrside_LH)

        self.cam_thread.rrsideRH_pitch_updated.connect(self._set_button_color_rrside_RH)
        self.cam_thread.rrsideRH_numofPart_updated.connect(self._set_numlabel_text_rrside_RH)

        

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
        button_P5902A509 = main_widget.findChild(QPushButton, "P5902A509button")
        button_P5902A510 = main_widget.findChild(QPushButton,"P5902A510button")

        self.siostatus = main_widget.findChild(QLabel, "status_sio")

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
        if button_P5902A509:
            button_P5902A509.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
            button_P5902A509.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 6))
        if button_P5902A510:
            button_P5902A510.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(7))
            button_P5902A510.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 7))


        # add extra widgets here

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

        # Widget 3 = generate training data
        button_takeimage = self.stackedWidget.widget(3).findChild(QPushButton, "takeimagebutton")
        button_takeimage.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "capture", True))

        button_takeimage_readwarp = self.stackedWidget.widget(3).findChild(QPushButton, "button_readwarp")
        label_takeimage_readwarp = self.stackedWidget.widget(3).findChild(QLabel, "label_readwarpcolor")
        button_takeimage_readwarp.pressed.connect(lambda: self._toggle_param_and_update_label("takeimage_readwarp", label_takeimage_readwarp))

        # Widget 4 = check carucoimage
        button_takearucoimage = self.stackedWidget.widget(4).findChild(QPushButton, "takeimagebutton")

        # frame = process_for_edge_detection(frame, self.slider_value)
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

        # Widget 5 6 7 
        button_rtwarp5 = self.stackedWidget.widget(5).findChild(QPushButton, "rtwarpbutton")
        label_rtwarp5 = self.stackedWidget.widget(5).findChild(QLabel, "rtwarpcolor")
        button_rtwarp5.pressed.connect(lambda: self._toggle_param_and_update_label("rtwarp", label_rtwarp5))

        button_savewarp5 = self.stackedWidget.widget(5).findChild(QPushButton, "savewarpbutton")
        button_savewarp5.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "savewarp", True))


        # Monitoring the kansei and furyou. Need to streamline this later
        self.kanseihin_number_cowltop = self.stackedWidget.widget(5).findChild(QLabel, "status_kansei")
        self.furyouhin_number_cowltop = self.stackedWidget.widget(5).findChild(QLabel, "status_furyou")

        self.kanseihin_number_rrside_lh = self.stackedWidget.widget(6).findChild(QLabel, "status_kansei")
        self.furyouhin_number_rrside_lh = self.stackedWidget.widget(6).findChild(QLabel, "status_furyou")

        self.kanseihin_number_rrside_rh = self.stackedWidget.widget(7).findChild(QLabel, "status_kansei")
        self.furyouhin_number_rrside_rh = self.stackedWidget.widget(7).findChild(QLabel, "status_furyou")


        #-> kensain name
        # self.kensain_name = self.stackedWidget.widget(5).findChild(QLineEdit, "kensain_name")
        # self.kensain_name.textChanged.connect(lambda: self._set_cam_params(self.cam_thread, "kensainName", self.kensain_name.text()))

        self.connect_line_edit_text_changed(widget_index=5, line_edit_name="kensain_name", cam_param="kensainName")
        self.connect_line_edit_text_changed(widget_index=6, line_edit_name="kensain_name", cam_param="kensainName")
        self.connect_line_edit_text_changed(widget_index=7, line_edit_name="kensain_name", cam_param="kensainName")


        # add "b" button as shortcut for button_kensa
        self.shortcut_kensa = QShortcut(QKeySequence("b"), self)
        self.shortcut_kensa.activated.connect(self.simulateButtonKensaClicks)

        # add "n" button as shortcut for button_rekensa
        # self.shortcut_rekensa = QShortcut(QKeySequence("n"), self)
        # self.shortcut_rekensa.activated.connect(self.button_rekensa.click)

        # Widget 6 -> HOOD RR SIDE LH 5902A509

        # Connect kensa button for widget 5 6 7

        self.button_kensa5 = self.stackedWidget.widget(5).findChild(QPushButton, "kensaButton")
        self.button_kensa6 = self.stackedWidget.widget(6).findChild(QPushButton, "kensaButton")
        self.button_kensa7 = self.stackedWidget.widget(7).findChild(QPushButton, "kensaButton")
        
        self.button_kensa5.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "cowltop_doInspect", True))
        self.button_kensa6.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "cowltop_doInspect", True))
        self.button_kensa7.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "cowltop_doInspect", True))

        #Connect reset counter for widget 5 6 7
        self.connect_camparam_button(5, "counterReset", "resetCounter", True)
        self.connect_camparam_button(6, "counterReset", "resetCounter", True)
        self.connect_camparam_button(7, "counterReset", "resetCounter", True)

        #Connect deletewarp for widget 5 6 7
        self.connect_camparam_button(5, "delwarpbutton", "delwarp", True)
        self.connect_camparam_button(6, "delwarpbutton", "delwarp", True)
        self.connect_camparam_button(7, "delwarpbutton", "delwarp", True)

        #Connect savewarp for widget 5 6 7
        self.connect_camparam_button(5, "savewarpbutton", "savewarp", True)
        self.connect_camparam_button(6, "savewarpbutton", "savewarp", True)
        self.connect_camparam_button(7, "savewarpbutton", "savewarp", True)

        #Connect rekensa for widget 5 6 7
        self.connect_camparam_button(5, "rekensaButton", "cowltop_doReinspect", True)
        self.connect_camparam_button(6, "rekensaButton", "cowltop_doReinspect", True)
        self.connect_camparam_button(7, "rekensaButton", "cowltop_doReinspect", True)

        #Connect sio status for widget 5 6 7 
        self.siostatus_cowltop = [self.stackedWidget.widget(i).findChild(QLabel, "status_sio") for i in [0, 5, 6, 7]] # 0 is main page



        # _____________________________________________________________________________________________________
       # Find and connect quit buttons and main menu buttons in all widgets
        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            button_quit = widget.findChild(QPushButton, "quitbutton")
            button_main_menu = widget.findChild(QPushButton, "mainmenubutton")

            if button_quit:
                button_quit.clicked.connect(self._close_app)

            if button_main_menu:
                button_main_menu.clicked.connect(
                    lambda: self.stackedWidget.setCurrentIndex(0))
                button_main_menu.clicked.connect(
                    lambda: self._set_cam_params(self.cam_thread, 'widget', 0))
                # checkaruco.clicked.connect(lambda: set_params(self.cam_thread, 'check_aruco', False))

        self.stackedWidget.currentChanged.connect(self._on_widget_changed)

        self.setCentralWidget(self.stackedWidget)
        self.showFullScreen()


    def connect_line_edit_text_changed(self, widget_index, line_edit_name, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        line_edit = widget.findChild(QLineEdit, line_edit_name)
        if line_edit:
            line_edit.textChanged.connect(lambda text: self._set_cam_params(self.cam_thread, cam_param, text))


    def connect_camparam_button(self, widget_index, button_name, cam_param, value):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, button_name)
        if button:
            button.pressed.connect(lambda: self._set_cam_params(self.cam_thread, cam_param, value))


    def simulateButtonKensaClicks(self):
        # Simulate clicking multiple buttons
        self.button_kensa5.click()
        self.button_kensa6.click()
        self.button_kensa7.click()

    def _on_widget_changed(self, idx: int):
        if idx == 5 or idx == 6 or idx == 7:
            #Change widget value to equal to index of stacked widget first
            self._set_cam_params(self.cam_thread, 'widget', idx)
            self.cam_thread.initialize_model()
            

    def _close_app(self):
        self.cam_thread.stop()
        self.server_monitor_thread.stop()
        QCoreApplication.instance().quit()

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
        for i in [5, 6, 7]:
            widget = self.stackedWidget.widget(i)
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

    # def _cowltop_update_label(self, param, pitchvalue, labels):
    #     pitch = getattr(self.cam_thread.cam_config, pitchvalue)
    #     self._set_cam_params(self.cam_thread, param, True)
    #     colorOK = "green"
    #     colorNG = "red"
    #     # if the pitchvalue[i] is 1, then labels[i] is green, else red
    #     for i in range(len(pitch)):
    #         color = colorOK if pitch[i] else colorNG
    #         labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _set_button_color(self, pitch_data):
        colorOK = "green"
        colorNG = "red"

        label_names = ["P1color", "P2color", "P3color",
                       "P4color", "P5color", "Lsuncolor"]
        labels = [self.stackedWidget.widget(5).findChild(QLabel, name) for name in label_names]
        for i, pitch_value in enumerate(pitch_data):
            color = colorOK if pitch_value else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _set_button_color_rrside_LH(self, pitch_data): #For rr side, consists of 6 pitches and Lsun (total Length)
        colorOK = "green"
        colorNG = "red"
        # print (pitch_data)
        label_names = ["P1color", "P2color", "P3color",
                       "P4color", "P5color", "P6color", 
                       "Lsuncolor"]
        
        labels = [self.stackedWidget.widget(6).findChild( QLabel, name) for name in label_names]
        
        for i, pitch_value in enumerate(pitch_data):
            color = colorOK if pitch_value else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _set_button_color_rrside_RH(self, pitch_data): #For rr side, consists of 6 pitches and Lsun (total Length)
        colorOK = "green"
        colorNG = "red"

        label_names = ["P1color", "P2color", "P3color",
                       "P4color", "P5color", "P6color", 
                       "Lsuncolor"]
        
        labels = [self.stackedWidget.widget(7).findChild( QLabel, name) for name in label_names]
        
        for i, pitch_value in enumerate(pitch_data):
            color = colorOK if pitch_value else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")


    def _set_numlabel_text(self, numofPart):
        self.kanseihin_number_cowltop.setText(str(numofPart[0]))
        self.furyouhin_number_cowltop.setText(str(numofPart[1]))

    def _set_numlabel_text_rrside_LH(self, numofPart):
        self.kanseihin_number_rrside_lh.setText(str(numofPart[0]))
        self.furyouhin_number_rrside_lh.setText(str(numofPart[1]))

    def _set_numlabel_text_rrside_RH(self, numofPart):
        self.kanseihin_number_rrside_rh.setText(str(numofPart[0]))
        self.furyouhin_number_rrside_rh.setText(str(numofPart[1]))

def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
