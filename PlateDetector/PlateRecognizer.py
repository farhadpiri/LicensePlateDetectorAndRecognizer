class PlateRecognizer():
    def __init__(self):
        pass

    def plate_segmentor(self,image):
        pass

    def recognize(self,plates_dets,image):
        for plate in plates_dets:
            for *xyxy, conf, cls in reversed(plate):
                plate_image = image[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                segments = self.plate_segmentor(plate_image)

