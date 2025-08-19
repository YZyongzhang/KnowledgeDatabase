from DataImpl import Data_impl
from Window.win import Windows
from PyQt6.QtWidgets import QApplication
import sys



def handle_data(data):
    print(data)
if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = Windows()
    window.show()
    window.summit_data.connect(handle_data)
    app.exec()


    
