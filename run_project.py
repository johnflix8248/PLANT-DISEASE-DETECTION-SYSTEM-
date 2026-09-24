"""
Spyder entry point: download dataset -> train CNN -> open detection GUI.
Run this file (F5) after:  pip install -r requirements.txt
"""
from download_dataset import download_dataset
from train_cnn import main as train_main
from predict_gui import App


if __name__ == "__main__":
    download_dataset()
    train_main()
    App().mainloop()
