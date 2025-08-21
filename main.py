from Window.win import Windows
from PyQt6.QtWidgets import QApplication
import sys
from DataBase.DataImpl import data_impl



def handle_data(data):
    data_impl.input(data=data)

if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = Windows()
    window.show()
    window.summit_data.connect(handle_data)
    app.exec()
    data_impl.save(path=None)


    
