import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication , QLabel , QMainWindow , QTextEdit , QMenu

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = QLabel("Click in this window")
        self.setCentralWidget(self.label)
    
    def mouseMoveEvent(self, a0):
        self.label.setText("mouseMoveEvent")
        
        if a0.button() == Qt.MouseButton.LeftButton:
            self.label.setText("leftmouse")
        
        elif a0.button() == Qt.MouseButton.MiddleButton:
            self.label.setText("midmouse")
        
        elif a0.button() == Qt.MouseButton.RightButton:
            self.label.setText("rightmouse")

    
    def mousePressEvent(self, a0):
        self.label.setText("mousePressEvent")

    def mouseReleaseEvent(self, a0):
        self.label.setText("mouseReleaseEvent")
    def mouseDoubleClickEvent(self, a0):
        self.label.setText("mouseDoubleClickEvent")


    # right mouse press excute this function
    def contextMenuEvent(self, event):
        context = QMenu(self)
        context.addAction(QAction("test 1 " , self))
        context.addAction(QAction("test 2 " , self))
        context.exec(event.globalPos())


app = QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()