import os
import sys
import shutil
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
from glob import glob
from tqdm import tqdm
import warnings
from datetime import datetime
import json

os.environ["QT_LOGGING_RULES"] = (
    "qt.gui.icc.debug=false;qt.gui.icc.warning=false;qt.gui.icc.info=false"
)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from PySide6.QtCore import (
    Qt,
    QObject,
    Signal,
    Slot,
    QRunnable,
    QThreadPool,
)
from PySide6.QtGui import (
    QFont,
    QStandardItemModel,
    QStandardItem,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QProgressBar,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QTextEdit,
    QListView,
    QMessageBox,
    QGridLayout,
    QTabWidget,
    QCheckBox,
)

from insightface.app import FaceAnalysis

DEFAULT_DATASET_PATH = "Dataset"
DEFAULT_MODEL_NAME = "face_model.pkl"
DEFAULT_MODEL_FOLDER = "Model/"
DEFAULT_EMBEDDING_MODEL = "buffalo_l"
DEFAULT_WINDOW_WIDTH = 1250
DEFAULT_WINDOW_HEIGHT = 950
DEFAULT_WINDOW_TITLE = "InsightFace Trainer"

EMBEDDING_MODELS = {
    "buffalo_l": {
        "name": "Buffalo L",
        "description": "Large model (275 MB) - High accuracy, for production use",
        "size": "275 MB",
        "recommended": True,
    },
    "buffalo_s": {
        "name": "Buffalo S",
        "description": "Small model (122 MB) - Faster processing, for real-time applications",
        "size": "122 MB",
        "recommended": False,
    },
    "buffalo_sc": {
        "name": "Buffalo SC",
        "description": "Super Compact model - Lightest and fastest processing",
        "size": "14.3 MB",
        "recommended": False,
    },
    "antelopev2": {
        "name": "Antelope V2",
        "description": "Latest model (344 MB) - Best performance and accuracy",
        "size": "344 MB",
        "recommended": False,
    },
}

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".avif")


class WorkerSignals(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)
    result = Signal(object)


class FaceVerificationWorker(QRunnable):
    def __init__(self, dataset_path, embedding_model_name):
        super().__init__()
        self.dataset_path = dataset_path
        self.embedding_model_name = embedding_model_name
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.signals.status.emit("Initializing face detection model...")

            app = FaceAnalysis(name=self.embedding_model_name)
            app.prepare(ctx_id=0)

            self.signals.status.emit("Scanning dataset folders...")

            person_folders = [
                d
                for d in os.listdir(self.dataset_path)
                if os.path.isdir(os.path.join(self.dataset_path, d)) and d != "Invalid"
            ]

            if not person_folders:
                self.signals.error.emit("No person folders found in dataset!")
                return

            dataset_parent_dir = os.path.dirname(self.dataset_path)
            invalid_folder = os.path.join(dataset_parent_dir, "Invalid")
            os.makedirs(invalid_folder, exist_ok=True)

            total_images = 0
            for person_name in person_folders:
                person_dir = os.path.join(self.dataset_path, person_name)
                image_files = []
                for ext in VALID_IMAGE_EXTENSIONS:
                    image_files.extend(glob(os.path.join(person_dir, f"*{ext}")))
                total_images += len(image_files)

            processed_images = 0
            moved_images = 0
            results = {}

            for person_name in person_folders:
                person_dir = os.path.join(self.dataset_path, person_name)

                image_files = []
                for ext in VALID_IMAGE_EXTENSIONS:
                    image_files.extend(glob(os.path.join(person_dir, f"*{ext}")))

                valid_count = 0
                invalid_count = 0

                self.signals.status.emit(f"Verifying faces for {person_name}...")

                for img_path in image_files:
                    try:
                        img = np.array(Image.open(img_path).convert("RGB"))
                        faces = app.get(img)

                        if not faces:
                            invalid_person_dir = os.path.join(
                                invalid_folder, person_name
                            )
                            os.makedirs(invalid_person_dir, exist_ok=True)

                            new_path = os.path.join(
                                invalid_person_dir, os.path.basename(img_path)
                            )
                            shutil.move(img_path, new_path)
                            invalid_count += 1
                            moved_images += 1
                        elif len(faces) > 1:
                            invalid_person_dir = os.path.join(
                                invalid_folder, person_name
                            )
                            os.makedirs(invalid_person_dir, exist_ok=True)

                            new_path = os.path.join(
                                invalid_person_dir, os.path.basename(img_path)
                            )
                            shutil.move(img_path, new_path)
                            invalid_count += 1
                            moved_images += 1
                        else:
                            valid_count += 1

                    except Exception as e:
                        invalid_person_dir = os.path.join(invalid_folder, person_name)
                        os.makedirs(invalid_person_dir, exist_ok=True)

                        new_path = os.path.join(
                            invalid_person_dir, os.path.basename(img_path)
                        )
                        try:
                            shutil.move(img_path, new_path)
                            invalid_count += 1
                            moved_images += 1
                        except:
                            pass

                    processed_images += 1
                    progress = int((processed_images / total_images) * 100)
                    self.signals.progress.emit(progress)

                results[person_name] = {"valid": valid_count, "invalid": invalid_count}

            self.signals.status.emit(
                f"Face verification completed! Moved {moved_images} invalid images."
            )

            summary = {
                "results": results,
                "total_moved": moved_images,
                "total_processed": processed_images,
            }
            self.signals.result.emit(summary)
            self.signals.finished.emit()

        except Exception as e:
            self.signals.error.emit(f"Error during face verification: {str(e)}")


class TrainingWorker(QRunnable):
    def __init__(self, dataset_path, model_output_path, embedding_model_name):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_output_path = model_output_path
        self.embedding_model_name = embedding_model_name
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.signals.status.emit("Initializing embedding model...")
            self.signals.progress.emit(0)

            app = FaceAnalysis(name=self.embedding_model_name)
            app.prepare(ctx_id=0)

            self.signals.status.emit("Loading dataset and extracting embeddings...")
            self.signals.progress.emit(10)

            X, y = self.load_dataset_embeddings(app)

            if len(X) == 0:
                self.signals.error.emit("No valid training data found!")
                return

            self.signals.status.emit(f"Loaded {len(X)} training samples")
            self.signals.progress.emit(60)

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            self.signals.status.emit("Training SVM classifier...")
            self.signals.progress.emit(70)

            svm_classifier = SVC(kernel="linear", probability=True)
            svm_classifier.fit(X, y_encoded)

            self.signals.status.emit("Saving model...")
            self.signals.progress.emit(90)

            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            joblib.dump((svm_classifier, label_encoder), self.model_output_path)
            info_json_path = os.path.join(os.path.dirname(self.model_output_path), "info.json")

            training_info = {
                "model_file": os.path.basename(self.model_output_path),
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_persons": len(label_encoder.classes_),
                "total_samples": len(X),
                "person_names": list(label_encoder.classes_),
                "embedding_model": self.embedding_model_name,
                "dataset_path": self.dataset_path,
            }

            with open(info_json_path, "w", encoding="utf-8") as f:
                json.dump(training_info, f, indent=2, ensure_ascii=False)

            self.signals.status.emit(f"Model and training info saved successfully")
            self.signals.progress.emit(100)
            self.signals.finished.emit()

        except Exception as e:
            self.signals.error.emit(f"Error during training: {str(e)}")

    def load_dataset_embeddings(self, app):
        X = []
        y = []

        person_folders = [
            d
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d)) and d != "Invalid"
        ]

        total_folders = len(person_folders)

        for i, person_name in enumerate(person_folders):
            person_dir = os.path.join(self.dataset_path, person_name)

            image_files = []
            for ext in VALID_IMAGE_EXTENSIONS:
                image_files.extend(glob(os.path.join(person_dir, f"*{ext}")))

            for img_path in image_files:
                embedding = self.extract_embedding(app, img_path)
                if embedding is not None:
                    X.append(embedding)
                    y.append(person_name)

            folder_progress = int(((i + 1) / total_folders) * 50) + 10
            self.signals.progress.emit(folder_progress)
            self.signals.status.emit(
                f"Processing {person_name}... ({i+1}/{total_folders})"
            )

        return np.array(X), np.array(y)

    def extract_embedding(self, app, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            faces = app.get(img_np)

            if not faces:
                return None

            return faces[0].embedding
        except:
            return None


class ModernTrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.setGeometry(100, 100, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #3c3c3c;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #606060;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                color: #ffffff;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #707070;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
                border-color: #404040;
            }
            QLineEdit, QComboBox {
                background-color: #404040;
                border: 1px solid #606060;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
                min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #0078d4;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                padding: 8px;
            }
            QListView {
                background-color: #2d2d2d;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                padding: 4px;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                background-color: #2d2d2d;
                text-align: center;
                color: #ffffff;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
            QLabel {
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #3c3c3c;
            }
            QTabBar::tab {
                background-color: #404040;
                border: 1px solid #606060;
                padding: 4px 12px;
                margin-right: 2px;
                color: #ffffff;
                min-height: 12px;
                max-height: 24px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                border-color: #106ebe;
            }
            QTabBar::tab:hover:!selected {
                background-color: #505050;
            }
        """
        )

        self.thread_pool = QThreadPool()

        self.dataset_path = DEFAULT_DATASET_PATH
        self.model_output_path = DEFAULT_MODEL_NAME
        self.selected_embedding_model = DEFAULT_EMBEDDING_MODEL
        self.custom_model_path = ""
        self.faces_verified = False
        self.dataset_stats = {}

        self.init_ui()
        self.refresh_dataset_stats()

    def init_ui(self):
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)

        self.training_tab = QWidget()
        self.logs_tab = QWidget()

        self.main_tabs.addTab(self.training_tab, "🚀 Training")
        self.main_tabs.addTab(self.logs_tab, "📋 Logs")

        self.init_training_tab()
        self.init_logs_tab()

    def init_training_tab(self):
        layout = QVBoxLayout(self.training_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title_label = QLabel(DEFAULT_WINDOW_TITLE)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        config_grid = QGridLayout()

        dataset_group = self.create_dataset_config_group()
        config_grid.addWidget(dataset_group, 0, 0)

        model_group = self.create_model_config_group()
        config_grid.addWidget(model_group, 1, 0)

        output_group = self.create_output_config_group()
        config_grid.addWidget(output_group, 0, 1)

        action_group = self.create_action_buttons_group()
        config_grid.addWidget(action_group, 1, 1)

        layout.addLayout(config_grid)

        progress_group = QGroupBox("📊 Training Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { height: 25px; font-weight: bold; }"
        )
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to start training")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "QLabel { color: #cccccc; font-style: italic; margin: 5px; }"
        )
        progress_layout.addWidget(self.status_label)

        layout.addWidget(progress_group)

        stats_group = QGroupBox("📈 Dataset Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.total_count_label = QLabel("Total: 0 images in 0 folders")
        self.total_count_label.setStyleSheet(
            "QLabel { font-weight: bold; margin: 5px 0px 0px 0px; }"
        )
        stats_layout.addWidget(self.total_count_label)

        self.invalid_count_label = QLabel("Invalid: 0 images")
        self.invalid_count_label.setStyleSheet(
            "QLabel { font-weight: bold; margin: 0px 0px 5px 0px; }"
        )
        stats_layout.addWidget(self.invalid_count_label)

        self.dataset_list = QListView()
        self.dataset_model = QStandardItemModel()
        self.dataset_list.setModel(self.dataset_model)
        stats_layout.addWidget(self.dataset_list)

        layout.addWidget(stats_group)

    def create_dataset_config_group(self):
        group = QGroupBox("📁 Dataset Configuration")
        layout = QVBoxLayout(group)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Dataset Folder:"))
        self.dataset_path_input = QLineEdit(self.dataset_path)
        self.dataset_path_input.setPlaceholderText(
            "Select folder containing person subfolders..."
        )
        path_layout.addWidget(self.dataset_path_input)
        self.browse_dataset_btn = QPushButton("Browse")
        self.browse_dataset_btn.clicked.connect(self.browse_dataset_folder)
        path_layout.addWidget(self.browse_dataset_btn)
        layout.addLayout(path_layout)

        self.refresh_btn = QPushButton("🔄 Refresh Dataset Stats")
        self.refresh_btn.clicked.connect(self.refresh_dataset_stats)
        layout.addWidget(self.refresh_btn)

        return group

    def create_model_config_group(self):
        group = QGroupBox("🧠 Model Configuration")
        layout = QVBoxLayout(group)

        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Embedding Model:"))

        self.embedding_combo = QComboBox()
        for model_key, model_info in EMBEDDING_MODELS.items():
            display_text = f"{model_info['name']} ({model_info['size']})"
            if model_info["recommended"]:
                display_text += " ⭐"
            self.embedding_combo.addItem(display_text, model_key)

        for i in range(self.embedding_combo.count()):
            if "Buffalo L" in self.embedding_combo.itemText(i):
                self.embedding_combo.setCurrentIndex(i)
                break

        self.embedding_combo.currentIndexChanged.connect(
            self.on_embedding_model_changed
        )
        model_layout.addWidget(self.embedding_combo)

        self.model_description_label = QLabel()
        self.model_description_label.setWordWrap(True)
        self.model_description_label.setStyleSheet(
            "QLabel { color: #cccccc; font-size: 11px; font-style: italic; }"
        )
        self.update_model_description()
        model_layout.addWidget(self.model_description_label)

        layout.addLayout(model_layout)

        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("Custom Model:"))
        self.custom_model_input = QLineEdit()
        self.custom_model_input.setPlaceholderText(
            "Optional: Custom embedding model path..."
        )
        custom_layout.addWidget(self.custom_model_input)
        self.browse_custom_model_btn = QPushButton("Browse")
        self.browse_custom_model_btn.clicked.connect(self.browse_custom_model)
        custom_layout.addWidget(self.browse_custom_model_btn)
        layout.addLayout(custom_layout)

        return group

    def create_output_config_group(self):
        group = QGroupBox("💾 Output Configuration")
        layout = QVBoxLayout(group)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Model Output Folder:"))
        self.model_output_folder = QLineEdit(DEFAULT_MODEL_FOLDER)
        self.model_output_folder.setPlaceholderText(DEFAULT_MODEL_FOLDER)
        folder_layout.addWidget(self.model_output_folder)
        layout.addLayout(folder_layout)

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Model Output Name:"))
        self.model_output_input = QLineEdit(self.model_output_path)
        self.model_output_input.setPlaceholderText("face_model.pkl")
        output_layout.addWidget(self.model_output_input)
        layout.addLayout(output_layout)

        info_label = QLabel("Model will be saved in the specified folder.")
        info_label.setStyleSheet(
            "QLabel { color: #cccccc; font-size: 10px; font-style: italic; }"
        )
        layout.addWidget(info_label)

        return group

    def create_action_buttons_group(self):
        group = QGroupBox("⚡ Actions")
        layout = QVBoxLayout(group)

        self.verify_faces_btn = QPushButton("🔍 Verify Faces")
        self.verify_faces_btn.clicked.connect(self.start_face_verification)
        layout.addWidget(self.verify_faces_btn)

        self.skip_verification_checkbox = QCheckBox(
            "Skip Face Verification (I've already verified faces)"
        )
        self.skip_verification_checkbox.stateChanged.connect(
            self.on_skip_verification_changed
        )
        layout.addWidget(self.skip_verification_checkbox)

        self.start_training_btn = QPushButton("🚀 Start Training")
        self.start_training_btn.clicked.connect(self.start_training)
        self.start_training_btn.setEnabled(False)
        layout.addWidget(self.start_training_btn)

        return group

    def init_logs_tab(self):
        layout = QVBoxLayout(self.logs_tab)
        layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Process Logs & Information")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        log_group = QGroupBox("📋 Process Log")
        log_layout = QVBoxLayout(log_group)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_output)

        clear_btn = QPushButton("🗑️ Clear Logs")
        clear_btn.clicked.connect(self.log_output.clear)
        log_layout.addWidget(clear_btn)

        layout.addWidget(log_group)

    def browse_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", self.dataset_path
        )
        if folder:
            self.dataset_path = folder
            self.dataset_path_input.setText(folder)
            self.refresh_dataset_stats()
            self.faces_verified = False
            self.skip_verification_checkbox.setChecked(False)
            self.update_training_button_state()

    def browse_custom_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom Embedding Model",
            "",
            "Model files (*.onnx *.pkl);;All files (*.*)",
        )
        if file_path:
            self.custom_model_input.setText(file_path)
            self.custom_model_path = file_path

    def on_skip_verification_changed(self):
        if self.skip_verification_checkbox.isChecked():
            self.faces_verified = True
            self.log_message("✅ Face verification skipped by user")
        else:
            self.faces_verified = False
            self.log_message("ℹ️ Face verification required before training")

        self.update_training_button_state()

    def on_embedding_model_changed(self):
        model_key = self.embedding_combo.currentData()
        self.selected_embedding_model = model_key
        self.update_model_description()
        self.faces_verified = False
        self.skip_verification_checkbox.setChecked(False)
        self.update_training_button_state()

    def update_model_description(self):
        model_key = self.selected_embedding_model
        if model_key in EMBEDDING_MODELS:
            description = EMBEDDING_MODELS[model_key]["description"]
            self.model_description_label.setText(description)

    def refresh_dataset_stats(self):
        self.log_message("🔄 Refreshing dataset statistics...")

        if not os.path.exists(self.dataset_path):
            self.log_message(f"❌ Dataset path does not exist: {self.dataset_path}")
            return

        self.dataset_model.clear()
        self.dataset_stats = {}

        try:
            person_folders = [
                d
                for d in os.listdir(self.dataset_path)
                if os.path.isdir(os.path.join(self.dataset_path, d))
            ]

            dataset_parent_dir = os.path.dirname(self.dataset_path)
            invalid_folder_path = os.path.join(dataset_parent_dir, "Invalid")
            invalid_total = 0
            if os.path.exists(invalid_folder_path):
                invalid_subfolders = [
                    d
                    for d in os.listdir(invalid_folder_path)
                    if os.path.isdir(os.path.join(invalid_folder_path, d))
                ]
                for subfolder in invalid_subfolders:
                    subfolder_path = os.path.join(invalid_folder_path, subfolder)
                    for ext in VALID_IMAGE_EXTENSIONS:
                        invalid_total += len(
                            glob(os.path.join(subfolder_path, f"*{ext}"))
                        )

            total_images = 0
            valid_folders = 0

            for person_name in sorted(person_folders):
                person_dir = os.path.join(self.dataset_path, person_name)
                image_count = 0
                for ext in VALID_IMAGE_EXTENSIONS:
                    image_count += len(glob(os.path.join(person_dir, f"*{ext}")))

                self.dataset_stats[person_name] = image_count
                total_images += image_count
                valid_folders += 1

                item_text = f"👤 {person_name}: {image_count} images"
                item = QStandardItem(item_text)
                self.dataset_model.appendRow(item)

            self.total_count_label.setText(
                f"📊 Total: {total_images} images in {valid_folders} folders"
            )
            
            if invalid_total > 0:
                self.invalid_count_label.setText(f"❌ Invalid: {invalid_total} images (moved to invalid folders)")
                self.invalid_count_label.setVisible(True)
            else:
                self.invalid_count_label.setText("❌ Invalid: 0 images")
                self.invalid_count_label.setVisible(False)

            self.log_message(
                f"✅ Found {valid_folders} folders with {total_images} total images"
            )
            if invalid_total > 0:
                self.log_message(f"⚠️ Found {invalid_total} invalid images")

        except Exception as e:
            self.log_message(f"❌ Error reading dataset: {str(e)}")
            self.total_count_label.setText("❌ Error loading dataset")
            self.invalid_count_label.setText("❌ Error loading dataset")
            self.invalid_count_label.setVisible(True)

    def start_face_verification(self):
        if not os.path.exists(self.dataset_path):
            QMessageBox.warning(self, "Error", "Dataset folder does not exist!")
            return

        embedding_model = (
            self.custom_model_path
            if self.custom_model_path
            else self.selected_embedding_model
        )

        self.log_message("🔍 Starting face verification...")
        self.set_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        worker = FaceVerificationWorker(self.dataset_path, embedding_model)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.status.connect(self.update_status)
        worker.signals.finished.connect(self.on_face_verification_finished)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.result.connect(self.on_face_verification_result)

        self.thread_pool.start(worker)

    def on_face_verification_finished(self):
        self.log_message("✅ Face verification completed!")
        self.faces_verified = True
        self.skip_verification_checkbox.setChecked(False)
        self.update_training_button_state()
        self.set_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        self.refresh_dataset_stats()

    def on_face_verification_result(self, summary):
        self.log_message("📊 Verification Results:")
        results = summary.get("results", {})
        total_moved = summary.get("total_moved", 0)
        total_processed = summary.get("total_processed", 0)

        for person_name, counts in results.items():
            self.log_message(
                f"   {person_name}: {counts['valid']} valid, {counts['invalid']} invalid"
            )

        self.log_message(
            f"📈 Summary: {total_processed} images processed, {total_moved} moved to Invalid folder"
        )
        self.log_message(
            "ℹ️ Invalid images include: no face detected, multiple faces detected, or corrupted files"
        )

    def start_training(self):
        if not self.faces_verified:
            QMessageBox.warning(self, "Warning", "Please verify faces first!")
            return

        if not os.path.exists(self.dataset_path):
            QMessageBox.warning(self, "Error", "Dataset folder does not exist!")
            return

        model_name = self.model_output_input.text() or DEFAULT_MODEL_NAME
        if not model_name.endswith(".pkl"):
            model_name += ".pkl"

        output_folder = self.model_output_folder.text() or DEFAULT_MODEL_FOLDER
        if not output_folder.endswith("/"):
            output_folder += "/"

        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), output_folder, model_name
        )

        embedding_model = (
            self.custom_model_path
            if self.custom_model_path
            else self.selected_embedding_model
        )

        self.log_message(f"🚀 Starting training with model: {embedding_model}")
        self.log_message(f"💾 Output path: {model_path}")

        self.set_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        worker = TrainingWorker(self.dataset_path, model_path, embedding_model)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.status.connect(self.update_status)
        worker.signals.finished.connect(self.on_training_finished)
        worker.signals.error.connect(self.on_worker_error)

        self.thread_pool.start(worker)

    def on_training_finished(self):
        self.log_message("🎉 Training completed successfully! ✅")

        model_name = self.model_output_input.text() or DEFAULT_MODEL_NAME
        if not model_name.endswith(".pkl"):
            model_name += ".pkl"

        output_folder = self.model_output_folder.text() or DEFAULT_MODEL_FOLDER
        if not output_folder.endswith("/"):
            output_folder += "/"

        info_json_file = "info.json"

        self.log_message(f"💾 Model saved: {model_name}")
        self.log_message(f"📋 Training info (JSON): {info_json_file}")
        self.log_message(f"📁 Output folder: {output_folder}")

        self.set_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.information(
            self,
            "Success",
            f"Model training completed successfully! 🎉\n\n"
            f"Files saved:\n"
            f"• {model_name} (Model)\n"
            f"• {info_json_file} (Training Info)",
        )

    def on_worker_error(self, error_msg):
        self.log_message(f"❌ Error: {error_msg}")
        self.set_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", error_msg)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, status):
        self.status_label.setText(status)
        self.log_message(f"ℹ️ {status}")

    def update_training_button_state(self):
        self.start_training_btn.setEnabled(self.faces_verified)
        if self.faces_verified:
            self.start_training_btn.setStyleSheet("")
        else:
            self.start_training_btn.setStyleSheet("")

    def set_buttons_enabled(self, enabled):
        self.browse_dataset_btn.setEnabled(enabled)
        self.browse_custom_model_btn.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)
        self.verify_faces_btn.setEnabled(enabled)
        self.skip_verification_checkbox.setEnabled(enabled)
        if enabled:
            self.update_training_button_state()
        else:
            self.start_training_btn.setEnabled(False)

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_output.setTextCursor(cursor)


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "qt.gui.icc.debug=false;qt.gui.icc.warning=false"
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings, True)

    window = ModernTrainingWindow()
    window.show()

    sys.exit(app.exec())
