from PyQt6.QtWidgets import QApplication,QWidget,QPushButton,QMainWindow,QLabel,QLineEdit , QVBoxLayout
from PyQt6.QtCore import QSize , Qt
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.button_is_checked = True
        self.setWindowTitle("my app")
        self.button = QPushButton("Press Me!")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.the_button_was_clicked) # 当点击button的时候调用该函数。
        self.button.clicked.connect(self.the_button_was_toggled) # 调用该函数，绑定button的状态。点击开启为true，点击关闭为Flase
        self.button.setChecked(self.button_is_checked)
        # 后续可调用self.button.isChecked()获取状态
        self.setFixedSize(QSize(400 , 300))
        self.setCentralWidget(self.button)

    def the_button_was_clicked(self):
        self.button.setText("chilck!")
        self.button.setEnabled(False)
        self.setWindowTitle("My App!")
        print("Clicked!")

    def the_button_was_toggled(self , checked):
        self.button_is_checked = checked
        print("Checked？" , checked)
# 每一个文件中唯一的app event，需要传入sys.argv参数用来获取命令行的参数
app = QApplication(sys.argv)

# 创建一个窗口
# window = QWidget()
# window = QPushButton("push me!")
window = MainWindow()
# 展示这个窗口，如果没执行show，默认为false
window.show()

# 开始这个事件的loop
app.exec()