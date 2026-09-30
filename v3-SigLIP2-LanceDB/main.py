import sys
import os
import json
import shutil
import subprocess
from typing import Optional, List, Dict

import send2trash

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QSpinBox, QComboBox, QLineEdit, QGroupBox, QSplitter,
    QFrame, QScrollArea, QSizePolicy, QToolButton, QTabWidget, QDialog,
    QSlider, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QPixmap, QDesktopServices, QFont, QIcon, QColor, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QSize

from engine import (
    SigLIP2Engine, LanceDBManager, IndexManager,
    SUPPORTED_MODELS, DEFAULT_MODEL_KEY
)

# -------------------------------------------------------------------
# MODERN DARK STYLESHEET (ENGLISH)
# -------------------------------------------------------------------
DARK_STYLESHEET = """
QMainWindow {
    background-color: #111318;
}
QWidget {
    background-color: #111318;
    color: #E1E6ED;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #282D3A;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: bold;
    color: #388BFD;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    background-color: #111318;
}
QPushButton {
    background-color: #1B1F2A;
    color: #E1E6ED;
    border: 1px solid #2D3446;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #242A3B;
    border-color: #388BFD;
}
QPushButton:pressed {
    background-color: #153A60;
}
QPushButton:disabled {
    background-color: #151821;
    color: #485264;
    border-color: #1E2330;
}
QPushButton#PrimaryBtn {
    background-color: #1F6FEB;
    color: #FFFFFF;
    border: 1px solid #388BFD;
}
QPushButton#PrimaryBtn:hover {
    background-color: #2679FC;
}
QPushButton#PrimaryBtn:pressed {
    background-color: #1656BD;
}
QPushButton#AccentBtn {
    background-color: #238636;
    color: #FFFFFF;
    border: 1px solid #2EA043;
}
QPushButton#AccentBtn:hover {
    background-color: #2EA043;
}
QPushButton#DangerBtn {
    background-color: #842029;
    color: #FFFFFF;
    border: 1px solid #B02A37;
}
QPushButton#DangerBtn:hover {
    background-color: #B02A37;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #171B24;
    border: 1px solid #2A3142;
    border-radius: 6px;
    padding: 6px 10px;
    color: #FFFFFF;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #388BFD;
}
QProgressBar {
    border: 1px solid #2A3142;
    border-radius: 6px;
    text-align: center;
    background-color: #171B24;
    color: #FFFFFF;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1F6FEB, stop:1 #238636);
    border-radius: 5px;
}
QListWidget {
    background-color: #141720;
    border: 1px solid #242936;
    border-radius: 8px;
    padding: 6px;
}
QListWidget::item {
    background-color: #1A1F2C;
    border: 1px solid #283042;
    border-radius: 8px;
    margin-bottom: 8px;
    padding: 4px;
}
QListWidget::item:hover {
    border-color: #388BFD;
    background-color: #202738;
}
QTabWidget::pane {
    border: 1px solid #282D3A;
    border-radius: 8px;
    background-color: #141720;
    padding: 10px;
}
QTabBar::tab {
    background-color: #191D28;
    color: #8B949E;
    border: 1px solid #282D3A;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background-color: #141720;
    color: #58A6FF;
    border-color: #388BFD;
}
QLabel#HeaderTitle {
    font-size: 22px;
    font-weight: bold;
    color: #FFFFFF;
}
QLabel#SubTitle {
    font-size: 12px;
    color: #8B949E;
}
"""

# -------------------------------------------------------------------
# WORKER THREADS
# -------------------------------------------------------------------
class EngineInitWorker(QThread):
    finished_signal = pyqtSignal(object, str)

    def __init__(self, model_key: str):
        super().__init__()
        self.model_key = model_key

    def run(self):
        try:
            engine = SigLIP2Engine(model_key=self.model_key)
            self.finished_signal.emit(engine, "")
        except Exception as e:
            self.finished_signal.emit(None, str(e))


class IndexingWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(int, str)

    def __init__(self, index_manager: IndexManager, root_folder: str, batch_size: int = 32):
        super().__init__()
        self.index_manager = index_manager
        self.root_folder = root_folder
        self.batch_size = batch_size

    def run(self):
        try:
            count = self.index_manager.index_directory(
                self.root_folder,
                batch_size=self.batch_size,
                progress_callback=lambda cur, tot, fname: self.progress_signal.emit(cur, tot, fname)
            )
            self.finished_signal.emit(count, "")
        except Exception as e:
            self.finished_signal.emit(0, str(e))


class SearchWorker(QThread):
    finished_signal = pyqtSignal(list, str)

    def __init__(
        self,
        engine: SigLIP2Engine,
        db_manager: LanceDBManager,
        mode: str,  # "text", "image", "composed"
        query_content: str,
        composed_text: str,
        top_k: int,
        category: str,
        subcategory: str
    ):
        super().__init__()
        self.engine = engine
        self.db_manager = db_manager
        self.mode = mode
        self.query_content = query_content
        self.composed_text = composed_text
        self.top_k = top_k
        self.category = category
        self.subcategory = subcategory

    def run(self):
        try:
            if self.mode == "text":
                query_vec = self.engine.encode_text_query(self.query_content)
            elif self.mode == "image":
                query_vec = self.engine.encode_images_batch([self.query_content])[0]
            elif self.mode == "composed":
                query_vec = self.engine.encode_composed_query(
                    image_path=self.query_content,
                    text_modification=self.composed_text
                )
            else:
                raise ValueError(f"Unknown search mode: {self.mode}")

            results = self.db_manager.search_vector(
                query_vector=query_vec,
                top_k=self.top_k,
                category=self.category,
                subcategory=self.subcategory,
                is_text_search=(self.mode == "text")
            )
            for r in results:
                r["search_mode"] = self.mode

            self.finished_signal.emit(results, "")
        except Exception as e:
            self.finished_signal.emit([], str(e))


class DuplicateSearchWorker(QThread):
    finished_signal = pyqtSignal(list, str)

    def __init__(self, db_manager: LanceDBManager, threshold: float = 0.95):
        super().__init__()
        self.db_manager = db_manager
        self.threshold = threshold

    def run(self):
        try:
            pairs = self.db_manager.find_duplicates(similarity_threshold=self.threshold)
            self.finished_signal.emit(pairs, "")
        except Exception as e:
            self.finished_signal.emit([], str(e))

# -------------------------------------------------------------------
# LIGHTBOX FULLSCREEN PREVIEW DIALOG
# -------------------------------------------------------------------
class LightboxViewerDialog(QDialog):
    def __init__(self, results: List[Dict], current_index: int, parent=None):
        super().__init__(parent)
        self.results = results
        self.current_index = current_index
        self.setWindowTitle("Image Preview Lightbox")
        self.resize(1050, 720)
        self.setStyleSheet("background-color: #0D0F14; color: #FFFFFF;")

        self.init_ui()
        self.show_image(self.current_index)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #050608; border: 1px solid #1F2430; border-radius: 8px;")
        layout.addWidget(self.lbl_image, stretch=1)

        info_row = QHBoxLayout()
        self.lbl_info = QLabel("Filename | Cosine Similarity: 0.000")
        self.lbl_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #58A6FF;")
        info_row.addWidget(self.lbl_info)
        info_row.addStretch()

        self.lbl_counter = QLabel("1 / 1")
        self.lbl_counter.setStyleSheet("font-size: 13px; color: #8B949E;")
        info_row.addWidget(self.lbl_counter)
        layout.addLayout(info_row)

        nav_row = QHBoxLayout()
        btn_prev = QPushButton("◀ Previous (Left Arrow)")
        btn_prev.setFocusPolicy(Qt.NoFocus)
        btn_prev.clicked.connect(self.prev_image)
        nav_row.addWidget(btn_prev)

        btn_open_img = QPushButton("🖼️ Open File")
        btn_open_img.setFocusPolicy(Qt.NoFocus)
        btn_open_img.clicked.connect(self.open_current_file)
        nav_row.addWidget(btn_open_img)

        btn_open_folder = QPushButton("📂 Open Folder")
        btn_open_folder.setFocusPolicy(Qt.NoFocus)
        btn_open_folder.clicked.connect(self.open_current_folder)
        nav_row.addWidget(btn_open_folder)

        btn_next = QPushButton("Next (Right Arrow) ▶")
        btn_next.setFocusPolicy(Qt.NoFocus)
        btn_next.clicked.connect(self.next_image)
        nav_row.addWidget(btn_next)

        layout.addLayout(nav_row)
        self.setLayout(layout)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_A):
            self.prev_image()
            event.accept()
        elif event.key() in (Qt.Key_Right, Qt.Key_D):
            self.next_image()
            event.accept()
        elif event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
        else:
            super().keyPressEvent(event)

    def show_image(self, index: int):
        if not self.results or index < 0 or index >= len(self.results):
            return

        self.current_index = index
        res = self.results[index]
        path = res["path"]
        raw_cos = res.get("raw_cosine", res.get("score", 0.0) / 100.0)

        if os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaled(self.lbl_image.size() - QSize(20, 20), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_image.setPixmap(scaled)
            else:
                self.lbl_image.setText("Unable to load image")
        else:
            self.lbl_image.setText("File not found")

        fname = os.path.basename(path)
        self.lbl_info.setText(f"📄 {fname}  |  Cosine Similarity: {raw_cos:.3f}")
        self.lbl_counter.setText(f"{index + 1} of {len(self.results)}")

    def prev_image(self):
        if self.current_index > 0:
            self.show_image(self.current_index - 1)

    def next_image(self):
        if self.current_index < len(self.results) - 1:
            self.show_image(self.current_index + 1)

    def open_current_file(self):
        if self.results and os.path.exists(self.results[self.current_index]["path"]):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.results[self.current_index]["path"]))

    def open_current_folder(self):
        if self.results and os.path.exists(self.results[self.current_index]["path"]):
            folder = os.path.dirname(self.results[self.current_index]["path"])
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

# -------------------------------------------------------------------
# VISUAL DUPLICATE CARD WIDGET WITH 1-CLICK TRASH ACTION
# -------------------------------------------------------------------
class DuplicateClusterCard(QWidget):
    cluster_resolved_signal = pyqtSignal()

    def __init__(self, cluster: Dict, db_manager: LanceDBManager, parent=None):
        super().__init__(parent)
        self.cluster = cluster
        self.db_manager = db_manager
        self.primary_path = cluster["primary_path"]
        self.primary_rel = cluster["primary_rel_path"]
        self.matches = cluster["matches"]
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        group = QGroupBox(f"📷 Primary Base Image: {os.path.basename(self.primary_path)}")
        g_layout = QVBoxLayout()

        # Primary Image Info
        p_row = QHBoxLayout()
        p_thumb = QLabel()
        p_thumb.setFixedSize(120, 80)
        p_thumb.setStyleSheet("border: 1px solid #388BFD; border-radius: 4px; background-color: #090B0E;")
        if os.path.exists(self.primary_path):
            pix = QPixmap(self.primary_path).scaled(120, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p_thumb.setPixmap(pix)
        p_row.addWidget(p_thumb)

        p_info = QVBoxLayout()
        p_info.addWidget(QLabel(f"<b>{os.path.basename(self.primary_path)}</b>"))
        p_info.addWidget(QLabel(f"Path: {self.primary_rel}"))
        p_badge = QLabel("KEEP (Primary File)")
        p_badge.setStyleSheet("background-color: #172B1E; color: #238636; font-weight: bold; border-radius: 4px; padding: 2px 6px;")
        p_info.addWidget(p_badge)
        p_row.addLayout(p_info, stretch=1)
        g_layout.addLayout(p_row)

        g_layout.addWidget(QLabel("<b>Duplicates Found:</b>"))

        # Duplicates Row List
        for dup in list(self.matches):
            d_path = dup["path"]
            d_rel = dup["rel_path"]
            sim = dup["similarity"]

            d_row = QHBoxLayout()
            d_thumb = QLabel()
            d_thumb.setFixedSize(100, 65)
            d_thumb.setStyleSheet("border: 1px solid #842029; border-radius: 4px; background-color: #090B0E;")
            if os.path.exists(d_path):
                dpix = QPixmap(d_path).scaled(100, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                d_thumb.setPixmap(dpix)
            d_row.addWidget(d_thumb)

            d_info = QVBoxLayout()
            d_info.addWidget(QLabel(f"<b>{os.path.basename(d_path)}</b> ({sim:.1f}% identical)"))
            d_info.addWidget(QLabel(f"Path: {d_rel}"))
            d_row.addLayout(d_info, stretch=1)

            btn_trash = QPushButton("🗑️ Trash Duplicate")
            btn_trash.setObjectName("DangerBtn")
            btn_trash.setToolTip("Safely moves this duplicate image file to system Trash and deletes it from LanceDB index.")
            btn_trash.clicked.connect(lambda chk, p=d_path: self.trash_duplicate_file(p))
            d_row.addWidget(btn_trash)

            g_layout.addLayout(d_row)

        group.setLayout(g_layout)
        layout.addWidget(group)
        self.setLayout(layout)

    def trash_duplicate_file(self, file_path: str):
        reply = QMessageBox.question(
            self,
            "Move File to Trash",
            f"Are you sure you want to move this duplicate file to system Trash?\n\n{file_path}",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                if os.path.exists(file_path):
                    send2trash.send2trash(file_path)
                self.db_manager.delete_record_by_path(file_path)
                QMessageBox.information(self, "Trashed", f"File moved to Trash successfully:\n{os.path.basename(file_path)}")
                self.cluster_resolved_signal.emit()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not move file to Trash:\n{e}")

# -------------------------------------------------------------------
# DUPLICATE FINDER DIALOG WITH VISUAL PREVIEWS & TRASH ACTIONS
# -------------------------------------------------------------------
class DuplicateFinderDialog(QDialog):
    def __init__(self, db_manager: LanceDBManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setWindowTitle("Duplicate & Near-Identical Image Finder")
        self.resize(950, 680)
        self.setStyleSheet("background-color: #111318; color: #FFFFFF;")

        self.init_ui()
        self.run_search()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel("🔍 Near-Duplicate Image Clusters (Cosine Similarity ≥ 95%)"))

        self.lbl_status = QLabel("Scanning LanceDB database for duplicate images...")
        self.lbl_status.setStyleSheet("color: #388BFD; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area, stretch=1)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

        self.setLayout(layout)

    def run_search(self):
        self.worker = DuplicateSearchWorker(self.db_manager, threshold=0.95)
        self.worker.finished_signal.connect(self.on_duplicates_found)
        self.worker.start()

    def on_duplicates_found(self, pairs: List[Dict], error_msg: str):
        if error_msg:
            self.lbl_status.setText(f"Error scanning duplicates: {error_msg}")
            return

        # Clear existing cards
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not pairs:
            self.lbl_status.setText("No duplicate image clusters found in the dataset.")
            return

        self.lbl_status.setText(f"Found {len(pairs)} duplicate clusters.")

        for cluster in pairs:
            card = DuplicateClusterCard(cluster, self.db_manager, parent=self)
            card.cluster_resolved_signal.connect(self.run_search)
            self.scroll_layout.addWidget(card)

# -------------------------------------------------------------------
# CUSTOM LIST RESULT CARD WIDGET (SINGLE COSIINE SIMILARITY BADGE)
# -------------------------------------------------------------------
class ResultCardWidget(QWidget):
    double_clicked_signal = pyqtSignal(int)

    def __init__(self, result: Dict, index: int):
        super().__init__()
        self.result = result
        self.index = index
        self.image_path = result["path"]
        self.score = result["score"]
        self.raw_cosine = result.get("raw_cosine", self.score / 100.0)
        self.search_mode = result.get("search_mode", "text")
        self.rel_path = result.get("rel_path", os.path.basename(self.image_path))
        self.category = result.get("category", "N/A")
        self.subcategory = result.get("subcategory", "N/A")
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(140, 100)
        self.thumb_label.setStyleSheet("border: 1px solid #2D3446; border-radius: 6px; background-color: #090B0E;")
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setCursor(Qt.PointingHandCursor)
        self.load_thumbnail()
        layout.addWidget(self.thumb_label)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        fname_label = QLabel(os.path.basename(self.image_path))
        fname_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #FFFFFF;")
        info_layout.addWidget(fname_label)

        path_label = QLabel(self.rel_path)
        path_label.setStyleSheet("color: #8B949E; font-size: 11px;")
        info_layout.addWidget(path_label)

        badges_layout = QHBoxLayout()
        badges_layout.setSpacing(6)

        # Dynamic badge coloring adapted to search mode
        if self.search_mode == "text":
            score_color = "#238636" if self.raw_cosine >= 0.08 else ("#1F6FEB" if self.raw_cosine >= 0.065 else "#8B949E")
        else:
            score_color = "#238636" if self.raw_cosine >= 0.70 else ("#D29922" if self.raw_cosine >= 0.45 else "#8B949E")

        score_badge = QLabel(f"Cosine Similarity: {self.raw_cosine:.3f}")
        score_badge.setStyleSheet(f"background-color: #172B1E; color: {score_color}; border: 1px solid {score_color}; font-weight: bold; border-radius: 4px; padding: 2px 6px; font-size: 11px;")
        badges_layout.addWidget(score_badge)

        cat_badge = QLabel(f"📁 {self.category} / {self.subcategory}")
        cat_badge.setStyleSheet("background-color: #1C2D42; color: #58A6FF; border-radius: 4px; padding: 2px 6px; font-size: 11px;")
        badges_layout.addWidget(cat_badge)

        badges_layout.addStretch()
        info_layout.addLayout(badges_layout)
        layout.addLayout(info_layout, stretch=1)

        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(6)

        btn_open_img = QPushButton("🖼️ Open Image")
        btn_open_img.setFixedWidth(130)
        btn_open_img.clicked.connect(self.open_image)
        btn_layout.addWidget(btn_open_img)

        btn_open_folder = QPushButton("📂 Open Folder")
        btn_open_folder.setFixedWidth(130)
        btn_open_folder.clicked.connect(self.open_folder)
        btn_layout.addWidget(btn_open_folder)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def load_thumbnail(self):
        if os.path.exists(self.image_path):
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(140, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.thumb_label.setPixmap(scaled)
                return
        self.thumb_label.setText("No Preview")

    def mouseDoubleClickEvent(self, event):
        self.double_clicked_signal.emit(self.index)
        super().mouseDoubleClickEvent(event)

    def open_image(self):
        if os.path.exists(self.image_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.image_path))

    def open_folder(self):
        if not os.path.exists(self.image_path):
            return
        if sys.platform == "win32":
            subprocess.run(["explorer", "/select,", os.path.normpath(self.image_path)])
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", self.image_path])
        else:
            folder = os.path.dirname(self.image_path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

# -------------------------------------------------------------------
# MAIN APPLICATION WINDOW (100% ENGLISH)
# -------------------------------------------------------------------
class MattexSearchAppV3(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mattex Image Search v3 — SigLIP 2 & LanceDB Engine")
        self.resize(1200, 820)

        self.dataset_root: Optional[str] = None
        self.db_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_lancedb")

        self.query_image_path: Optional[str] = None
        self.composed_image_path: Optional[str] = None

        self.engine: Optional[SigLIP2Engine] = None
        self.db_manager: Optional[LanceDBManager] = None
        self.index_manager: Optional[IndexManager] = None

        self.current_model_key = DEFAULT_MODEL_KEY
        self.current_results: List[Dict] = []

        self.init_ui()
        self.init_engine_async(self.current_model_key)

    def init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # ---------------- HEADER ----------------
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_lbl = QLabel("Mattex Image Search v3")
        title_lbl.setObjectName("HeaderTitle")
        subtitle_lbl = QLabel("Multimodal Search Engine with Google SigLIP 2 & LanceDB Vector Store")
        subtitle_lbl.setObjectName("SubTitle")
        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)
        header.addLayout(title_box)
        header.addStretch()

        # Tools & Settings Buttons with Tooltips
        btn_duplicates = QPushButton("🔍 Find Duplicates")
        btn_duplicates.setToolTip("Scans LanceDB database to detect duplicate or near-identical image pairs (≥95% similarity) and safely move duplicates to Trash.")
        btn_duplicates.clicked.connect(self.open_duplicate_finder)
        header.addWidget(btn_duplicates)

        self.btn_disk_index = QPushButton("⚙️ Build Disk Index")
        self.btn_disk_index.setToolTip("Builds an IVF-PQ (Inverted File Product Quantization) approximate index on disk.\nAccelerates vector search to sub-millisecond speeds for large datasets (10,000+ images) without increasing RAM usage.")
        self.btn_disk_index.clicked.connect(self.build_disk_index)
        header.addWidget(self.btn_disk_index)

        header.addWidget(QLabel("AI Model:"))
        self.cmb_model = QComboBox()
        for k, v in SUPPORTED_MODELS.items():
            self.cmb_model.addItem(v["description"], k)
        self.cmb_model.currentIndexChanged.connect(self.on_model_changed)
        header.addWidget(self.cmb_model)
        main_layout.addLayout(header)

        # ---------------- DATASET & INDEXING SECTION ----------------
        group_index = QGroupBox("1. Dataset Folder & Indexing")
        index_layout = QVBoxLayout()
        index_layout.setSpacing(8)

        top_idx_row = QHBoxLayout()
        self.btn_select_folder = QPushButton("📁 1. Select Dataset Folder")
        self.btn_select_folder.clicked.connect(self.select_dataset_folder)
        top_idx_row.addWidget(self.btn_select_folder)

        self.lbl_folder_path = QLabel("No dataset folder selected")
        self.lbl_folder_path.setStyleSheet("color: #8B949E; font-style: italic;")
        top_idx_row.addWidget(self.lbl_folder_path, stretch=1)

        self.btn_start_index = QPushButton("🚀 2. Start Indexing")
        self.btn_start_index.setObjectName("PrimaryBtn")
        self.btn_start_index.clicked.connect(self.run_indexing)
        top_idx_row.addWidget(self.btn_start_index)
        index_layout.addLayout(top_idx_row)

        prog_row = QHBoxLayout()
        self.pb_index = QProgressBar()
        self.pb_index.setValue(0)
        prog_row.addWidget(self.pb_index, stretch=1)

        self.lbl_status = QLabel("Status: Initializing AI engine...")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #58A6FF;")
        prog_row.addWidget(self.lbl_status)
        index_layout.addLayout(prog_row)

        group_index.setLayout(index_layout)
        main_layout.addWidget(group_index)

        # ---------------- SEARCH MODES TABS ----------------
        self.tabs_search = QTabWidget()
        self.tabs_search.setToolTip("Switch between Text Prompt Search, Visual Image Similarity Search, and Composed Multimodal Search.")

        # TAB 1: Text Search
        tab_text = QWidget()
        layout_text = QVBoxLayout()
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("🔍 Text Query:"))
        self.txt_query = QLineEdit()
        self.txt_query.setPlaceholderText("Type a prompt e.g., 'red sports car', 'cat on a sofa', 'snowy mountain landscape'...")
        self.txt_query.returnPressed.connect(self.run_text_search)
        t_row.addWidget(self.txt_query, stretch=1)

        self.btn_text_search = QPushButton("Search by Text")
        self.btn_text_search.setObjectName("PrimaryBtn")
        self.btn_text_search.clicked.connect(self.run_text_search)
        t_row.addWidget(self.btn_text_search)
        layout_text.addLayout(t_row)
        tab_text.setLayout(layout_text)

        # TAB 2: Image Search
        tab_image = QWidget()
        layout_image = QVBoxLayout()
        i_row = QHBoxLayout()
        i_row.addWidget(QLabel("🖼️ Query Image:"))

        self.lbl_img_preview = QLabel()
        self.lbl_img_preview.setFixedSize(110, 75)
        self.lbl_img_preview.setStyleSheet("border: 1px solid #388BFD; border-radius: 6px; background-color: #090B0E;")
        self.lbl_img_preview.setAlignment(Qt.AlignCenter)
        self.lbl_img_preview.setText("No Preview")
        i_row.addWidget(self.lbl_img_preview)

        self.lbl_img_path = QLabel("No image selected")
        self.lbl_img_path.setStyleSheet("color: #8B949E; font-weight: bold;")
        i_row.addWidget(self.lbl_img_path, stretch=1)

        btn_browse_img = QPushButton("Browse Image...")
        btn_browse_img.clicked.connect(self.select_query_image)
        i_row.addWidget(btn_browse_img)

        self.btn_img_search = QPushButton("Search Similar Images")
        self.btn_img_search.setObjectName("PrimaryBtn")
        self.btn_img_search.clicked.connect(self.run_image_search)
        i_row.addWidget(self.btn_img_search)
        layout_image.addLayout(i_row)
        tab_image.setLayout(layout_image)

        # TAB 3: Composed Search (Image + Text Modification)
        tab_composed = QWidget()
        layout_comp = QVBoxLayout()
        c_row = QHBoxLayout()
        c_row.addWidget(QLabel("🖼️ Base Image:"))

        self.lbl_comp_img_preview = QLabel()
        self.lbl_comp_img_preview.setFixedSize(110, 75)
        self.lbl_comp_img_preview.setStyleSheet("border: 1px solid #388BFD; border-radius: 6px; background-color: #090B0E;")
        self.lbl_comp_img_preview.setAlignment(Qt.AlignCenter)
        self.lbl_comp_img_preview.setText("No Preview")
        c_row.addWidget(self.lbl_comp_img_preview)

        self.lbl_comp_img_path = QLabel("No image selected")
        self.lbl_comp_img_path.setStyleSheet("color: #8B949E; font-weight: bold;")
        c_row.addWidget(self.lbl_comp_img_path, stretch=1)

        btn_browse_comp = QPushButton("Browse Base Image...")
        btn_browse_comp.clicked.connect(self.select_composed_image)
        c_row.addWidget(btn_browse_comp)
        layout_comp.addLayout(c_row)

        c_row2 = QHBoxLayout()
        c_row2.addWidget(QLabel("✏️ Text Modification:"))
        self.txt_composed_mod = QLineEdit()
        self.txt_composed_mod.setPlaceholderText("e.g. 'in blue color', 'outdoor background', 'modern style'...")
        c_row2.addWidget(self.txt_composed_mod, stretch=1)

        self.btn_composed_search = QPushButton("Search Composed")
        self.btn_composed_search.setObjectName("AccentBtn")
        self.btn_composed_search.clicked.connect(self.run_composed_search)
        c_row2.addWidget(self.btn_composed_search)
        layout_comp.addLayout(c_row2)
        tab_composed.setLayout(layout_comp)

        self.tabs_search.addTab(tab_text, "🔍 Text Search")
        self.tabs_search.addTab(tab_image, "🖼️ Visual Search")
        self.tabs_search.addTab(tab_composed, "🎨 Composed Search (Image + Text)")
        main_layout.addWidget(self.tabs_search)

        # ---------------- FILTERS & CONTROLS ROW ----------------
        filter_row = QHBoxLayout()
        filter_row.setSpacing(12)

        filter_row.addWidget(QLabel("Category:"))
        self.cmb_cat = QComboBox()
        self.cmb_cat.addItem("All")
        self.cmb_cat.currentIndexChanged.connect(self.on_category_changed)
        filter_row.addWidget(self.cmb_cat)

        filter_row.addWidget(QLabel("Subcategory:"))
        self.cmb_subcat = QComboBox()
        self.cmb_subcat.addItem("All")
        filter_row.addWidget(self.cmb_subcat)

        filter_row.addWidget(QLabel("Top-K Results:"))
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 500)
        self.spin_topk.setValue(12)
        filter_row.addWidget(self.spin_topk)

        filter_row.addStretch()

        self.btn_export = QPushButton("💾 Export Results")
        self.btn_export.setToolTip("Export top search result images to a destination directory.")
        self.btn_export.clicked.connect(self.export_results)
        filter_row.addWidget(self.btn_export)

        main_layout.addLayout(filter_row)

        # ---------------- RESULTS HEADER & LIST ----------------
        results_header = QHBoxLayout()
        results_header.addWidget(QLabel("📋 Search Results"))
        self.lbl_result_count = QLabel("0 items found")
        self.lbl_result_count.setStyleSheet("color: #8B949E; font-weight: bold;")
        results_header.addWidget(self.lbl_result_count)
        results_header.addStretch()
        results_header.addWidget(QLabel("(Tip: Double-click any result to open Lightbox preview)"))

        main_layout.addLayout(results_header)

        self.list_results = QListWidget()
        main_layout.addWidget(self.list_results, stretch=1)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    # ---------------- ENGINE & ASYNC INIT ----------------
    def init_engine_async(self, model_key: str):
        self.lbl_status.setText(f"Loading AI model '{model_key}'...")
        self.btn_start_index.setEnabled(False)
        self.btn_text_search.setEnabled(False)
        self.btn_img_search.setEnabled(False)

        self.init_worker = EngineInitWorker(model_key)
        self.init_worker.finished_signal.connect(self.on_engine_initialized)
        self.init_worker.start()

    def on_engine_initialized(self, engine: Optional[SigLIP2Engine], error_msg: str):
        if error_msg or engine is None:
            self.lbl_status.setText(f"Engine initialization error: {error_msg}")
            QMessageBox.critical(self, "Initialization Error", f"Failed to load SigLIP 2 model:\n{error_msg}")
            return

        self.engine = engine
        self.db_manager = LanceDBManager(db_path=self.db_path, vector_dim=self.engine.dim)
        self.index_manager = IndexManager(self.engine, self.db_manager)

        indexed_total = self.db_manager.count()
        self.lbl_status.setText(f"Ready ({self.engine.model_id} on {self.engine.device}) - {indexed_total} images indexed.")

        self.btn_start_index.setEnabled(True)
        self.btn_text_search.setEnabled(True)
        self.btn_img_search.setEnabled(True)

        self.refresh_categories()
        self.update_disk_index_status()

    def update_disk_index_status(self):
        """Updates top bar Disk Index button text, style badge, and tooltip based on LanceDB disk index presence."""
        if not hasattr(self, 'btn_disk_index') or not self.btn_disk_index:
            return

        if self.db_manager and self.db_manager.count() > 0:
            if self.db_manager.has_disk_index():
                self.btn_disk_index.setText("⚡ Disk Index: Active ✓")
                self.btn_disk_index.setStyleSheet("""
                    QPushButton {
                        background-color: #172B1E;
                        color: #3FB950;
                        border: 1px solid #238636;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #1F3A29;
                        border-color: #2EA043;
                    }
                """)
                self.btn_disk_index.setToolTip(
                    "✓ IVF-PQ Disk Index is ACTIVE on disk!\n"
                    "Vector search is fully accelerated for high performance.\n"
                    "Click to inspect status or rebuild index if dataset changed."
                )
            else:
                self.btn_disk_index.setText("⚙️ Build Disk Index")
                self.btn_disk_index.setStyleSheet("")
                self.btn_disk_index.setToolTip(
                    "No disk index found.\n"
                    "Builds an IVF-PQ approximate index on disk for datasets (256+ images) without increasing RAM usage."
                )
        else:
            self.btn_disk_index.setText("⚙️ Build Disk Index")
            self.btn_disk_index.setStyleSheet("")

    def on_model_changed(self, index: int):
        model_key = self.cmb_model.itemData(index)
        if model_key and model_key != self.current_model_key:
            reply = QMessageBox.question(
                self,
                "Change AI Model",
                "Switching vector models will reload the neural network weights.\nProceed?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.current_model_key = model_key
                self.init_engine_async(model_key)
            else:
                idx = self.cmb_model.findData(self.current_model_key)
                self.cmb_model.blockSignals(True)
                self.cmb_model.setCurrentIndex(idx)
                self.cmb_model.blockSignals(False)

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Dataset Root Directory")
        if folder:
            self.dataset_root = folder
            self.lbl_folder_path.setText(folder)

    def refresh_categories(self):
        if not self.db_manager:
            return
        cats, subcat_map = self.db_manager.get_categories_and_subcategories()
        self.subcat_map = subcat_map

        self.cmb_cat.blockSignals(True)
        self.cmb_cat.clear()
        self.cmb_cat.addItems(cats)
        self.cmb_cat.blockSignals(False)

        self.on_category_changed()

    def on_category_changed(self):
        cat = self.cmb_cat.currentText()
        subcats = self.subcat_map.get(cat, ["All"]) if hasattr(self, "subcat_map") else ["All"]
        self.cmb_subcat.clear()
        self.cmb_subcat.addItems(subcats)

    # ---------------- INDEXING WORKFLOW ----------------
    def run_indexing(self):
        if not self.dataset_root:
            return QMessageBox.warning(self, "No Dataset Folder", "Please select an image dataset folder first.")

        self.btn_start_index.setEnabled(False)
        self.pb_index.setValue(0)
        self.lbl_status.setText("Indexing in progress...")

        self.idx_worker = IndexingWorker(self.index_manager, self.dataset_root)
        self.idx_worker.progress_signal.connect(self.on_indexing_progress)
        self.idx_worker.finished_signal.connect(self.on_indexing_finished)
        self.idx_worker.start()

    def on_indexing_progress(self, current: int, total: int, filename: str):
        pct = int((current / total) * 100) if total > 0 else 0
        self.pb_index.setValue(pct)
        self.lbl_status.setText(f"Indexed {current}/{total}: {filename}")

    def on_indexing_finished(self, count: int, error_msg: str):
        self.btn_start_index.setEnabled(True)
        if error_msg:
            QMessageBox.critical(self, "Indexing Error", f"An error occurred during indexing:\n{error_msg}")
            self.lbl_status.setText("Error during indexing.")
        else:
            total_indexed = self.db_manager.count()
            msg = f"Indexing complete! {count} new images added." if count > 0 else "No new images to index."
            QMessageBox.information(self, "Indexing Complete", msg)
            self.lbl_status.setText(f"Ready - Total images in LanceDB: {total_indexed}")
            self.refresh_categories()
            self.update_disk_index_status()

    # ---------------- SEARCH WORKFLOW ----------------
    def select_query_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Query Image", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
        if path:
            self.query_image_path = path
            self.lbl_img_path.setText(os.path.basename(path))
            pix = QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaled(110, 75, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_img_preview.setPixmap(scaled)

    def select_composed_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Base Image", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
        if path:
            self.composed_image_path = path
            self.lbl_comp_img_path.setText(os.path.basename(path))
            pix = QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaled(110, 75, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_comp_img_preview.setPixmap(scaled)

    def run_text_search(self):
        query = self.txt_query.text().strip()
        if not query:
            return QMessageBox.warning(self, "Empty Query", "Please enter a text prompt to search.")
        self.execute_search(mode="text", query_content=query, composed_text="")

    def run_image_search(self):
        if not self.query_image_path or not os.path.exists(self.query_image_path):
            return QMessageBox.warning(self, "No Image Selected", "Please select a query image first.")
        self.execute_search(mode="image", query_content=self.query_image_path, composed_text="")

    def run_composed_search(self):
        if not self.composed_image_path or not os.path.exists(self.composed_image_path):
            return QMessageBox.warning(self, "No Base Image", "Please select a base query image.")
        mod = self.txt_composed_mod.text().strip()
        if not mod:
            return QMessageBox.warning(self, "No Text Modification", "Please enter a text modification prompt (e.g. 'in blue color').")
        self.execute_search(mode="composed", query_content=self.composed_image_path, composed_text=mod)

    def execute_search(self, mode: str, query_content: str, composed_text: str):
        if not self.engine or not self.db_manager:
            return

        self.btn_text_search.setEnabled(False)
        self.btn_img_search.setEnabled(False)
        self.btn_composed_search.setEnabled(False)
        self.lbl_status.setText("Executing vector search...")

        cat = self.cmb_cat.currentText()
        subcat = self.cmb_subcat.currentText()
        top_k = self.spin_topk.value()

        self.search_worker = SearchWorker(
            engine=self.engine,
            db_manager=self.db_manager,
            mode=mode,
            query_content=query_content,
            composed_text=composed_text,
            top_k=top_k,
            category=cat,
            subcategory=subcat
        )
        self.search_worker.finished_signal.connect(self.on_search_finished)
        self.search_worker.start()

    def on_search_finished(self, results: List[Dict], error_msg: str):
        self.btn_text_search.setEnabled(True)
        self.btn_img_search.setEnabled(True)
        self.btn_composed_search.setEnabled(True)
        self.lbl_status.setText("Search completed.")

        if error_msg:
            QMessageBox.critical(self, "Search Error", f"An error occurred during search:\n{error_msg}")
            return

        self.current_results = results
        self.list_results.clear()
        self.lbl_result_count.setText(f"{len(results)} items found")

        if not results:
            QMessageBox.information(self, "No Results", "No matching images found for the specified criteria.")
            return

        for idx, res in enumerate(results):
            item = QListWidgetItem()
            widget = ResultCardWidget(res, index=idx)
            widget.double_clicked_signal.connect(self.open_lightbox)
            item.setSizeHint(widget.sizeHint())
            self.list_results.addItem(item)
            self.list_results.setItemWidget(item, widget)

    def open_lightbox(self, index: int):
        if self.current_results:
            dialog = LightboxViewerDialog(self.current_results, current_index=index, parent=self)
            dialog.exec_()

    def open_duplicate_finder(self):
        if not self.db_manager or self.db_manager.count() == 0:
            return QMessageBox.warning(self, "Empty Database", "Index some images first before scanning for duplicates.")
        dialog = DuplicateFinderDialog(self.db_manager, parent=self)
        dialog.exec_()

    def build_disk_index(self):
        if not self.db_manager or self.db_manager.count() == 0:
            return QMessageBox.warning(self, "Empty Database", "Index some images first before building a disk index.")

        count = self.db_manager.count()
        has_idx = self.db_manager.has_disk_index()

        if count < 256:
            status_prefix = "✓ IVF-PQ Disk Index is ACTIVE on disk!\n\n" if has_idx else ""
            msg = (
                f"{status_prefix}"
                f"Your database currently contains {count} indexed image(s).\n\n"
                "Building a new IVF-PQ disk index requires at least 256 images in LanceDB.\n"
                "For smaller datasets (< 256 images), exact brute-force vector search runs in < 1 millisecond automatically."
            )
            return QMessageBox.information(self, "Disk Index Information", msg)

        if has_idx:
            reply = QMessageBox.question(
                self,
                "Disk Index Active — Rebuild?",
                "✓ An IVF-PQ Disk Index is currently ACTIVE on disk!\n\n"
                "Vector search is already fully accelerated.\n"
                "Would you like to rebuild/re-index it now?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        else:
            reply = QMessageBox.question(
                self,
                "Build Disk Index (IVF-PQ)",
                "Build an IVF-PQ approximate nearest neighbor index on disk for LanceDB?\n\n"
                "This accelerates vector search speeds for large datasets (10,000+ to 500,000+ images) without increasing RAM usage.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        ok = self.db_manager.create_disk_index()
        if ok:
            self.update_disk_index_status()
            QMessageBox.information(self, "Disk Index Built", "✓ IVF-PQ disk index created successfully and is now active.")

    def export_results(self):
        if not self.current_results:
            return QMessageBox.warning(self, "No Results", "Perform a search first to export results.")
        dest_folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder to Export Images")
        if dest_folder:
            copied = 0
            for r in self.current_results:
                src = r["path"]
                if os.path.exists(src):
                    dst = os.path.join(dest_folder, os.path.basename(src))
                    shutil.copy(src, dst)
                    copied += 1
            QMessageBox.information(self, "Export Complete", f"Successfully exported {copied} image files to:\n{dest_folder}")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MattexSearchAppV3()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
