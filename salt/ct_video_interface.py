"""
CT序列视频标注界面
基于PyQt5的图形用户界面，支持CT序列的视频分割标注
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QGraphicsView, QGraphicsScene, QPushButton, 
                           QRadioButton, QProgressBar, QTextEdit, QSplitter, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent, QFont
from PyQt5.QtCore import Qt, QRectF, QThread, pyqtSignal


class PropagationWorker(QThread):
    """视频传播工作线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)
    
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
    
    def run(self):
        """执行传播任务"""
        try:
            success = self.editor.propagate_in_video()
            self.finished.emit(success)
        except Exception as e:
            print(f"传播线程错误: {e}")
            self.finished.emit(False)


class CTGraphicsView(QGraphicsView):
    """CT图像显示视图"""
    
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        
        # 设置渲染选项
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 设置优化选项
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # 设置滚动条
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 设置变换锚点
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)
        
        # 创建场景
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = None
    
    def set_image(self, cv_image):
        """设置显示图像"""
        if cv_image is None:
            return
        
        # 转换OpenCV图像为QImage
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # 创建QPixmap
        pixmap = QPixmap.fromImage(q_image)
        
        if self.image_item:
            self.image_item.setPixmap(pixmap)
        else:
            self.image_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        old_pos = self.mapToScene(event.pos())
        
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        
        self.scale(zoom_factor, zoom_factor)
        
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标点击事件"""
        if not self.image_item:
            return
        
        # 获取点击位置
        pos = event.pos()
        pos_in_scene = self.mapToScene(pos)
        pos_in_item = pos_in_scene - self.image_item.pos()
        
        x, y = int(pos_in_item.x()), int(pos_in_item.y())
        
        # 检查点击是否在图像范围内
        if x < 0 or y < 0 or x >= self.image_item.pixmap().width() or y >= self.image_item.pixmap().height():
            return
        
        # 确定标签（左键正样本，右键负样本）
        if event.button() == Qt.LeftButton:
            label = 1  # 正样本
            click_type = "正样本"
        elif event.button() == Qt.RightButton:
            label = 0  # 负样本
            click_type = "负样本"
        else:
            return
        
        print(f"界面接收到{click_type}点击: 屏幕坐标({pos.x()}, {pos.y()}) -> 图像坐标({x}, {y})")
        
        # 添加点标注
        success = self.editor.add_point([x, y], label)
        if success:
            self.set_image(self.editor.display_image)
        else:
            print(f"添加{click_type}点失败")


class CTVideoInterface(QWidget):
    """CT序列视频标注主界面"""
    
    def __init__(self, app, editor):
        super().__init__()
        self.app = app
        self.editor = editor
        self.base_data_root = getattr(editor, 'base_data_root', None)
        
        # 设置窗口属性
        self.setWindowTitle("CT序列视频分割标注工具")
        self.setGeometry(100, 100, 1400, 900)
        
        # 初始化UI
        self._init_ui()
        
        # 初始化传播工作线程
        self.propagation_worker = None
        
        # 更新显示
        self._update_display()
        self._update_info()
    
    def _init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QHBoxLayout()
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：图像显示区域
        left_widget = self._create_image_area()
        splitter.addWidget(left_widget)
        
        # 右侧：控制面板
        right_widget = self._create_control_panel()
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setStretchFactor(0, 3)  # 图像区域占3/4
        splitter.setStretchFactor(1, 1)  # 控制面板占1/4
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def _create_image_area(self):
        """创建图像显示区域"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 顶部工具栏
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # 图像显示视图
        self.graphics_view = CTGraphicsView(self.editor)
        layout.addWidget(self.graphics_view)
        
        # 底部状态栏
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        widget.setLayout(layout)
        return widget
    
    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = QWidget()
        layout = QHBoxLayout()
        
        # 选择序列按钮
        self.choose_seq_btn = QPushButton("📂 选择序列")
        self.choose_seq_btn.clicked.connect(self.choose_sequence)
        layout.addWidget(self.choose_seq_btn)
        
        layout.addWidget(QLabel("|"))
        
        # 导航按钮
        self.prev_btn = QPushButton("◀ 上一帧")
        self.prev_btn.clicked.connect(self.prev_frame)
        layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("下一帧 ▶")
        self.next_btn.clicked.connect(self.next_frame)
        layout.addWidget(self.next_btn)
        
        layout.addWidget(QLabel("|"))
        
        # 标注操作按钮
        self.add_object_btn = QPushButton("➕ 添加对象")
        self.add_object_btn.clicked.connect(self.add_current_object)
        layout.addWidget(self.add_object_btn)
        
        self.reset_btn = QPushButton("重置当前帧")
        self.reset_btn.clicked.connect(self.reset_current_frame)
        layout.addWidget(self.reset_btn)
        
        self.propagate_btn = QPushButton("🎬 传播标注")
        self.propagate_btn.clicked.connect(self.propagate_annotations)
        layout.addWidget(self.propagate_btn)
        
        layout.addWidget(QLabel("|"))
        
        # 保存按钮
        self.save_btn = QPushButton("💾 保存")
        self.save_btn.clicked.connect(self.save_annotations)
        layout.addWidget(self.save_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        toolbar.setLayout(layout)
        return toolbar

    def choose_sequence(self):
        """选择新的CT序列目录并重新加载"""
        try:
            # 以当前序列的 images 目录为初始目录
            try:
                init_dir = str(self.editor.ct_loader.images_path)
            except Exception:
                init_dir = ""
            if not self.base_data_root:
                br = QFileDialog.getExistingDirectory(self, "请选择数据总路径（例如 F:\\qxfg\\export-image）")
                if br:
                    self.base_data_root = br
            selected_dir = QFileDialog.getExistingDirectory(self, "请选择包含PNG/JPG序列的目录", init_dir)
            if not selected_dir:
                return
            from pathlib import Path
            p = Path(selected_dir)

            sequence_name = p.name

            # ... existing code ...
            # 使用已有的编辑器复用模型，只重新加载序列与状态
            ok = self.editor.reload_sequence(
                images_path=str(p),
                base_data_root=self.base_data_root,
                sequence_name=sequence_name
            )
            if not ok:
                self.status_label.setText("选择序列失败：重载序列时出现错误")
                return

            # 图形视图绑定仍使用同一个 editor（此行可选，用于确保引用一致）
            if hasattr(self, 'graphics_view') and self.graphics_view is not None:
                self.graphics_view.editor = self.editor

            # 刷新显示与信息
            self._update_display()
            self._update_info()
            self.status_label.setText(f"已加载: {sequence_name}")
        except Exception as e:
            print(f"选择序列失败: {e}")
            self.status_label.setText(f"选择序列失败: {e}")
    
    def _create_control_panel(self):
        """创建控制面板"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 帧信息
        info_group = self._create_info_group()
        layout.addWidget(info_group)
        
        # 类别选择
        category_group = self._create_category_group()
        layout.addWidget(category_group)
        
        # 操作说明
        help_group = self._create_help_group()
        layout.addWidget(help_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def _create_info_group(self):
        """创建信息显示组"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("帧信息")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 信息显示
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
        widget.setLayout(layout)
        return widget
    
    def _create_category_group(self):
        """创建类别选择组"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("标注类别")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 类别单选按钮
        self.category_buttons = []
        for i, category in enumerate(self.editor.categories):
            btn = QRadioButton(category)
            if i == 0:
                btn.setChecked(True)
            btn.toggled.connect(lambda checked, cat_id=i: self._on_category_changed(cat_id, checked))
            layout.addWidget(btn)
            self.category_buttons.append(btn)
        
        widget.setLayout(layout)
        return widget
    
    def _create_help_group(self):
        """创建帮助说明组"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("操作说明")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 说明文本
        help_text = QTextEdit()
        help_text.setMaximumHeight(200)
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <b>鼠标操作：</b><br>
        • 左键点击：添加正样本点<br>
        • 右键点击：添加负样本点<br>
        • 滚轮：缩放图像<br><br>
        
        <b>标注流程：</b><br>
        1. 选择标注类别<br>
        2. 在关键帧上点击标注<br>
        3. 点击"传播标注"进行时序传播<br>
        4. 浏览其他帧检查结果<br>
        5. 保存标注结果<br><br>
        
        <b>快捷键：</b><br>
        • A/D：上一帧/下一帧<br>
        • R：重置当前帧<br>
        • Ctrl+S：保存<br>
        • ESC：退出
        """)
        layout.addWidget(help_text)
        
        widget.setLayout(layout)
        return widget
    
    def _update_display(self):
        """更新图像显示"""
        if self.editor.display_image is not None:
            self.graphics_view.set_image(self.editor.display_image)
    
    def _update_info(self):
        """更新信息显示"""
        info = self.editor.get_frame_info()
        
        info_text = f"""
            当前帧：{info['current_frame'] + 1} / {info['total_frames']}
            当前类别：{info['current_category']}
            关键帧：{'是' if info['is_keyframe'] else '否'}
            已标注：{'是' if info['has_annotations'] else '否'}
        """.strip()
        
        self.info_text.setPlainText(info_text)
        
        # 更新窗口标题
        self.setWindowTitle(f"CT序列视频分割标注工具 - 帧 {info['current_frame'] + 1}/{info['total_frames']}")
    
    def _on_category_changed(self, category_id, checked):
        """类别选择改变"""
        if checked:
            self.editor.set_category(category_id)
            self._update_info()
    
    def prev_frame(self):
        """上一帧"""
        self.editor.prev_frame()
        self._update_display()
        self._update_info()
    
    def next_frame(self):
        """下一帧"""
        self.editor.next_frame()
        self._update_display()
        self._update_info()
    
    def reset_current_frame(self):
        """重置当前帧"""
        self.editor.reset_current_frame()
        self._update_display()
        self._update_info()
        self.status_label.setText("当前帧已重置")
    
    def propagate_annotations(self):
        """传播标注"""
        if not self.editor.keyframes:
            self.status_label.setText("请先在关键帧上添加标注")
            return
        
        # 禁用按钮
        self.propagate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.status_label.setText("正在传播标注...")
        
        # 启动传播线程
        self.propagation_worker = PropagationWorker(self.editor)
        self.propagation_worker.finished.connect(self._on_propagation_finished)
        self.propagation_worker.start()
    
    def _on_propagation_finished(self, success):
        """传播完成回调"""
        self.propagate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_label.setText("标注传播完成")
            self._update_display()
            self._update_info()
        else:
            self.status_label.setText("标注传播失败")
    
    def add_current_object(self):
        """添加当前对象到标注"""
        success = self.editor.add_current_object()
        if success:
            self.status_label.setText("对象已添加到标注")
            self._update_display()
            self._update_info()
        else:
            self.status_label.setText("添加对象失败")
    
    def save_annotations(self):
        """保存标注"""
        success = self.editor.save_annotations()
        if success:
            self.status_label.setText("标注已保存")
        else:
            self.status_label.setText("保存失败")
    
    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key_Escape:
            self.app.quit()
        elif event.key() == Qt.Key_A:
            self.prev_frame()
        elif event.key() == Qt.Key_D:
            self.next_frame()
        elif event.key() == Qt.Key_R:
            self.reset_current_frame()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_annotations()
        else:
            super().keyPressEvent(event)