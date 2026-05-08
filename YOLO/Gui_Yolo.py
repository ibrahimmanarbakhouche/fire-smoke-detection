import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer
from ultralytics import YOLO


class FireSmokeDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.class_info = {
            0: {"name": "smoke", "color": (255, 0, 0)},
            1: {"name": "fire", "color": (0, 0, 255)}
        }

    def predict(self, img: np.ndarray) -> np.ndarray:
        results = self.model.predict(img, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = box.conf[0]
            info = self.class_info.get(cls, {"name": "unknown", "color": (0, 255, 0)})
            label = f"{info['name']} {conf:.2f}"
            color = info["color"]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img


class FireSmokeApp(QWidget):

    def __init__(self, detector: FireSmokeDetector):
        super().__init__()
        self.detector = detector
        self.setWindowTitle("Détection Feu / Fumée")
        self.setGeometry(100, 100, 800, 600)
        self._setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.cap = None

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.btn_load_image = QPushButton("Charger une image")
        self.btn_load_image.clicked.connect(self._load_image)
        layout.addWidget(self.btn_load_image)

        self.btn_webcam = QPushButton("Webcam en temps réel")
        self.btn_webcam.clicked.connect(self._start_webcam)
        layout.addWidget(self.btn_webcam)

        self.setLayout(layout)

    def _load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choisir une image", "", "Images (*.jpg *.png *.jpeg)"
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "Erreur", "Impossible de charger l'image.")
            return
        img = self.detector.predict(img)
        self._display_image(img)

    def _start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Erreur", "Impossible d'ouvrir la webcam.")
                return
            self.timer.start(30)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.detector.predict(frame)
            self._display_image(frame)

    def _display_image(self, img: np.ndarray):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(),
            self.image_label.height()
        ))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    # MODEL_PATH = "C:/Users/Ibrahim Manar/Desktop/DL Project V2/archive/YOLO_runs/dfire_model/weights/best.pt"
    MODEL_PATH = r"C:\Users\Ibrahim Manar\Desktop\DL Project V2\archive\YOLO_runs\dfire_model\weights\best.pt"
    detector = FireSmokeDetector(MODEL_PATH)
    app = QApplication(sys.argv)
    window = FireSmokeApp(detector)
    window.show()
    sys.exit(app.exec())