import numpy as np
import cv2
import os
from PlateDetector.plate_detector import PlateDtector
ROOT_DIR = os.getcwd()

if(__name__=="__main__"):
    Address = "Cars308.png"
    img = cv2.imread(Address)
    plate_detector = PlateDtector()
    plate_detector.detect_plate(img)