from ObjectDetector.YOLOv7.YOLO_Plate_Detector import YOLO_Plate_Detector
class PlateDtector:
    def __init__(self):
        self.plate_detector = YOLO_Plate_Detector()

    def detect_plate(self,image):

        plate_boxs = self.plate_detector.detect(image)


