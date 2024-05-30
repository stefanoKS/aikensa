from PyQt5.QtCore import QThread, pyqtSignal
import socket
import time


class TimeMonitorThread(QThread):
    time_signal = pyqtSignal(str)  

    def __init__(self, check_interval=10):
        super().__init__()
        self.check_interval = check_interval
        self.running = True

    def run(self):
        while self.running:
            current_time = time.strftime("%Y年%m月%d日 %H:%M")
            self.time_signal.emit(current_time)
            time.sleep(self.check_interval)

    def stop(self):
        self.running = False