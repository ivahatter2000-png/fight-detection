import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog,
                             QLabel, QVBoxLayout, QTextEdit)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QIcon  

MODEL_PATH = 'yolo_small_weights.pt'
THRESHOLD = 0.45
SAMPLE_INTERVAL = 30

class DetectionThread(QThread):
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(float, bool)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log_signal.emit("Ошибка: не удалось открыть видео")
            self.result_signal.emit(0.0, False)
            return

        confidences = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.log_signal.emit(f"Всего кадров: {total_frames}, шаг {SAMPLE_INTERVAL}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % SAMPLE_INTERVAL == 0:
                results = model(frame, verbose=False, imgsz=320)
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            if model.names[cls] == 'violence':
                                confidences.append(float(box.conf[0]))
            frame_count += 1
            if frame_count % (SAMPLE_INTERVAL * 10) == 0:
                self.log_signal.emit(f"Обработано кадров: {frame_count}/{total_frames}")

        cap.release()
        avg_conf = np.mean(confidences) if confidences else 0.0
        is_fight = bool(avg_conf >= THRESHOLD)
        self.log_signal.emit(f"Средняя уверенность: {avg_conf:.3f}")
        self.log_signal.emit(f"Результат: {'ДРАКА ОБНАРУЖЕНА' if is_fight else 'ДРАКИ НЕТ'}")
        self.result_signal.emit(avg_conf, is_fight)

class FightApp(QWidget):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Распознавание конфликтов на видео')
        self.setGeometry(200, 200, 600, 450)
        layout = QVBoxLayout()

        self.btn_select = QPushButton('Выбрать видео')
        self.btn_select.clicked.connect(self.select_video)
        layout.addWidget(self.btn_select)

        self.lbl_file = QLabel('Файл не выбран')
        layout.addWidget(self.lbl_file)

        self.btn_detect = QPushButton('Анализировать')
        self.btn_detect.clicked.connect(self.detect)
        self.btn_detect.setEnabled(False)
        layout.addWidget(self.btn_detect)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def select_video(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Выберите видео', '', 'Video files (*.mp4 *.avi *.mov *.mkv)'
        )
        if fname:
            self.video_path = fname
            self.lbl_file.setText(fname)
            self.btn_detect.setEnabled(True)
            self.log_area.append(f"Выбрано видео: {fname}")

    def detect(self):
        if not self.video_path:
            return
        self.btn_detect.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.log_area.append("Запуск анализа...")

        self.thread = DetectionThread(self.video_path)
        self.thread.log_signal.connect(self.log_area.append)
        self.thread.result_signal.connect(self.on_result)
        self.thread.start()

    def on_result(self, conf, is_fight):
        self.log_area.append("Анализ завершён")
        if is_fight:
            self.log_area.append("⚠️ ВНИМАНИЕ: ОБНАРУЖЕНА ДРАКА! ⚠️")
        else:
            self.log_area.append("Драка не обнаружена.")
        self.btn_detect.setEnabled(True)
        self.btn_select.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icon.ico'))   # <-- сюда, перед созданием окна
    window = FightApp()
    window.show()
    sys.exit(app.exec_())