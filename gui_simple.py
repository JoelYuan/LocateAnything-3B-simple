import sys
import os
import re
import torch
import warnings
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout,
    QDialog
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from transformers import AutoModel, AutoTokenizer, AutoProcessor

# 屏蔽无关警告
warnings.filterwarnings("ignore")

# ====================== 配置 ======================
MODEL_PATH = "/home/yuan/.cache/modelscope/hub/models/nv-community/LocateAnything-3B"
# ===================================================


class LocateAnythingWorker:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, fix_mistral_regex=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        # 关键：直接指定 device_map，避免后续设备移动冲突
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device,
            low_cpu_mem_usage=True
        ).eval()

    def detect(self, image, category):
        max_size = 1024
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        prompt = f"Locate all the instances that matches the description: {category}."
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        }]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        imgs, _ = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=imgs, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)

        with torch.no_grad():
            # 关键修复：强制 use_cache=True
            res = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=1024,
                generation_mode="hybrid",
                verbose=False,
                use_cache=True  # 强制启用缓存，解决该错误
            )
        return res[0] if isinstance(res, tuple) else res

    @staticmethod
    def parse_boxes(answer: str, img_w: int, img_h: int) -> list:
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = map(int, m.groups())
            boxes.append((
                x1 / 1000 * img_w,
                y1 / 1000 * img_h,
                x2 / 1000 * img_w,
                y2 / 1000 * img_h
            ))
        return boxes


# 模型加载 + 推理子线程
class WorkThread(QThread):
    finished_sig = pyqtSignal(object, list)
    error_sig = pyqtSignal(str)

    def __init__(self, img: Image.Image, target: str, parent=None):
        super().__init__(parent)
        self.img = img
        self.target = target
        self.worker = None

    def run(self):
        try:
            # 第一步：加载模型
            self.worker = LocateAnythingWorker(MODEL_PATH)
            # 第二步：执行检测
            raw_result = self.worker.detect(self.img, self.target)
            boxes = LocateAnythingWorker.parse_boxes(raw_result, self.img.width, self.img.height)
            self.finished_sig.emit(self.img, boxes)
        except Exception as e:
            self.error_sig.emit(str(e))


# 结果弹窗（展示带标注图片）
class ResultDialog(QDialog):
    def __init__(self, img: Image.Image, boxes: list, label: str, parent=None, original_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Detection Result")
        self.setFixedSize(900, 700)
        self.img = img
        self.boxes = boxes
        self.label_text = label
        self.original_path = original_path
        self.draw_img: Image.Image | None = None
        self.zoom_factor = 1.0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(800, 500)
        layout.addWidget(self.img_label)

        btn_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In (+)")
        self.zoom_out_btn = QPushButton("Zoom Out (-)")
        self.save_btn = QPushButton("Save Image")
        btn_layout.addWidget(self.zoom_in_btn)
        btn_layout.addWidget(self.zoom_out_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.save_btn.clicked.connect(self.save_image)

        self.draw_and_show()

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor + 0.25, 3.0)
        self.draw_and_show()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor - 0.25, 0.25)
        self.draw_and_show()

    def save_image(self):
        if self.draw_img is None:
            return
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.original_path:
            import os
            dir_name = os.path.dirname(self.original_path)
            base_name = os.path.splitext(os.path.basename(self.original_path))[0]
            new_name = f"{base_name}_{timestamp}.png"
            save_path = os.path.join(dir_name, new_name)
        else:
            save_path = f"detection_result_{timestamp}.png"
        self.draw_img.save(save_path)
        QMessageBox.information(self, "Saved", f"Image saved:\n{save_path}")

    def draw_and_show(self):
        self.draw_img = self.img.copy()
        draw = ImageDraw.Draw(self.draw_img)
        font = self._get_chinese_font(18)
        for (x1, y1, x2, y2) in self.boxes:
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
            draw.text((x1 + 4, y1 + 2), self.label_text, fill="#00FF00", font=font)

        w, h = self.draw_img.size
        q_img = QImage(self.draw_img.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        scaled_pix = pix.scaled(
            int(w * self.zoom_factor), int(h * self.zoom_factor),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.img_label.setPixmap(scaled_pix)

    def _get_chinese_font(self, size: int):
        chinese_fonts = [
            "NotoSansCJKsc-Regular.otf",
            "NotoSansSC-Regular.ttf",
            "WenQuanYiMicroHei.ttf",
            "SimHei.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        ]
        for font_path in chinese_fonts:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default(size=size)


# 主界面
class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocateAnything Detector")
        self.setFixedSize(800, 600)  # 4:3 固定比例
        self.raw_image: Image.Image | None = None
        self.raw_image_path: str | None = None
        self.work_thread: WorkThread | None = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # 左侧 3/4：图片预览区
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("background-color: #222; border: 1px solid #555;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.preview_label)
        main_layout.addWidget(left_panel, 3)

        # 右侧 1/4：控制面板
        right_panel = QWidget()
        right_panel.setFixedWidth(180)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.load_img_btn = QPushButton("Select Image")
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("Input target: car, person...")
        self.run_btn = QPushButton("Start Detection")

        right_layout.addWidget(QLabel("Target Object:"))
        right_layout.addWidget(self.target_edit)
        right_layout.addWidget(self.load_img_btn)
        right_layout.addWidget(self.run_btn)

        main_layout.addWidget(right_panel, 1)

        # 绑定信号
        self.load_img_btn.clicked.connect(self.select_image)
        self.run_btn.clicked.connect(self.start_detection)

    def select_image(self):
        """选择图片，立即预览，不加载模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        self.raw_image_path = file_path
        self.raw_image = Image.open(file_path).convert("RGB")
        self.show_preview(self.raw_image)

    def show_preview(self, img: Image.Image):
        """在主界面预览原图"""
        w, h = img.size
        q_img = QImage(img.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(pix)

    def start_detection(self):
        """点击按钮：加载模型 + 推理"""
        if self.raw_image is None:
            QMessageBox.warning(self, "Warning", "Please select an image first!")
            return
        target_text = self.target_edit.text().strip()
        if not target_text:
            QMessageBox.warning(self, "Warning", "Please input target object!")
            return

        # 禁用按钮，防止重复点击
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Loading Model...")
        QApplication.processEvents()

        # 启动子线程（加载模型+推理）
        self.work_thread = WorkThread(self.raw_image, target_text)
        self.work_thread.finished_sig.connect(self.on_detect_finish)
        self.work_thread.error_sig.connect(self.on_detect_error)
        self.work_thread.start()

    def on_detect_finish(self, img: Image.Image, boxes: list):
        """推理完成，弹出结果窗口"""
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Start Detection")
        target = self.target_edit.text().strip()
        result_dialog = ResultDialog(img, boxes, target, self, self.raw_image_path or "")
        result_dialog.exec()

    def on_detect_error(self, err_msg: str):
        """异常处理"""
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Start Detection")
        QMessageBox.critical(self, "Error", f"Task failed:\n{err_msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainGUI()
    window.show()
    sys.exit(app.exec())