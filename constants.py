from enum import Enum

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
]

WidgetUI.MAIN_PAGE.value