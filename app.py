"""
Главное приложение анализатора голоса (только эмоции).
"""
import sys
import os
import shutil
import sqlite3
import json
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QSplitter, QMessageBox, QFrame
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from src.features import extract_features
from src.model import predict_emotion, load_models
from src.visualize import create_plots

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
DB_PATH = BASE_DIR / "history.db"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def init_db():
    """Инициализация базы данных SQLite."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            original_filename TEXT,
            saved_audio_path TEXT,
            result_folder TEXT,
            emotion TEXT,
            emotion_confidence REAL
        )
    """)
    conn.commit()
    conn.close()


class HistoryTable(QTableWidget):
    """Виджет таблицы истории обработок."""
    def __init__(self):
        super().__init__()
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Время", "Файл", "Эмоция", "Детали"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.refresh()

    def refresh(self):
        """Загружает историю из БД."""
        self.setRowCount(0)
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, original_filename, emotion, result_folder
            FROM history ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            row_idx = self.rowCount()
            self.insertRow(row_idx)
            time_str = row[1][:16] if row[1] else ""
            self.setItem(row_idx, 0, QTableWidgetItem(time_str))
            self.setItem(row_idx, 1, QTableWidgetItem(row[2]))
            self.setItem(row_idx, 2, QTableWidgetItem(row[3]))
            btn = QPushButton("Открыть")
            btn.clicked.connect(lambda *args, r=row: self.open_result(r))
            self.setCellWidget(row_idx, 3, btn)

    def open_result(self, row_data):
        """Открывает папку с результатами в проводнике."""
        result_folder = row_data[4]
        if os.path.exists(result_folder):
            os.startfile(result_folder)
        else:
            QMessageBox.warning(self, "Ощибка", "Папка с результатами не найдена.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Emotion Analyzer")
        self.setMinimumSize(900, 600)

        try:
            load_models()
            self.models_loaded = True
        except FileNotFoundError:
            self.models_loaded = False

        self.init_ui()
        if not self.models_loaded:
            self.status_label.setText("Модели не найдены. Обучите модель в Jupyter и поместите в папку models/")
            self.analyze_btn.setEnabled(False)

    def init_ui(self):
        """Построение интерфейса."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Загрузить аудио и анализировать")
        self.analyze_btn.clicked.connect(self.analyze_audio)
        top_layout.addWidget(self.analyze_btn)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("История обработок"))
        self.history_table = HistoryTable()
        left_layout.addWidget(self.history_table)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.status_label = QLabel("Загрузите аудиофайл для начала анализа")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFrameStyle(QFrame.StyledPanel)
        right_layout.addWidget(self.image_label)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 600])
        main_layout.addWidget(splitter)

    def analyze_audio(self):
        """Обработка загруженного аудиофайла."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите аудиофайл", "", "Audio Files (*.wav *.mp3 *.ogg *.flac)")
        if not file_path:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(file_path).stem
        result_folder = RESULT_DIR / f"{timestamp}_{base_name}"
        result_folder.mkdir(parents=True, exist_ok=True)

        saved_audio = UPLOAD_DIR / f"{timestamp}_{Path(file_path).name}"
        shutil.copy2(file_path, saved_audio)

        self.status_label.setText("Анализирую... Ожидайте")
        QApplication.processEvents()

        try:
            features = extract_features(str(saved_audio))
            emotion, emo_conf = predict_emotion(features)
            plot_paths = create_plots(str(saved_audio), str(result_folder))

            report = {
                "emotion": emotion,
                "emotion_confidence": round(float(emo_conf), 3),
                "plots": plot_paths
            }
            with open(result_folder / "report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO history (timestamp, original_filename, saved_audio_path, result_folder,
                                     emotion, emotion_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                Path(file_path).name,
                str(saved_audio),
                str(result_folder),
                emotion,
                emo_conf
            ))
            conn.commit()
            conn.close()

            self.display_results(emotion, emo_conf, plot_paths)
            self.history_table.refresh()

        except Exception as e:
            print(f"ERROR: {e}")
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Ошибка")
            msg.setText("Не удалось обработать аудио:\n" + str(e))
            msg.setMinimumWidth(600)  # фиксируем ширину, чтобы кнопка «ОК» точно была видна
            msg.exec()
            self.status_label.setText("Ошибка анализа")

    def display_results(self, emotion, emo_conf, plot_paths):
        """Отображает результат анализа."""
        text = f"Эмоция: {emotion} (уверенность {emo_conf:.2%})"
        self.status_label.setText(text)

        mfcc_plot = [p for p in plot_paths if "mfcc" in p.lower()]
        if mfcc_plot:
            pixmap = QPixmap(mfcc_plot[0])
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.image_label.clear()


def main():
    init_db()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
