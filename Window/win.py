from PyQt6.QtWidgets import QApplication , QMainWindow , QTextEdit ,  QLayout , QPushButton , QVBoxLayout , QLabel , QWidget
from PyQt6.QtCore import pyqtSignal
import sys
class Windows(QMainWindow):
    summit_data = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        
        """
        定义一个window\n
        简单定义两个input和button\n
        一个input认为是key，另一个input认为是value\n
        点击button可以提交进入数据库\n
        """
        self.setWindowTitle("KonwleageDatabase")
        self.key_input = QTextEdit()
        self.value_input = QTextEdit()
        self.summit_button = QPushButton("提交")
        self.summit_button.clicked.connect(self._summit)
        self.win_layout = QVBoxLayout()
        self.win_layout.addWidget(QLabel("Key:"))
        self.win_layout.addWidget(self.key_input)
        self.win_layout.addWidget(QLabel("Value:"))
        self.win_layout.addWidget(self.value_input)
        self.win_layout.addWidget(self.summit_button)
        central = QWidget()
        central.setLayout(self.win_layout)
        self.setCentralWidget(central)

    def _summit(self):
        key = self.key_input.toPlainText()
        value = self.value_input.toPlainText()
        # print(f"key {key}: value {value}")
        self.key_input.clear()
        self.value_input.clear()
        data = {key:value}
        self.summit_data.emit(data)

