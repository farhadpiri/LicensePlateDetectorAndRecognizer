class ImageData:
    def __init__(self):
        self.image = None
        self.bounding_box = None

    def set_image(self,image):
        self.image = image

    def set_bounding_box(self,x_min,y_min,x_max,y_max):
        self.bounding_box = {"tl":(x_min,y_min),"bl":(x_min,y_max),"br":(x_max,y_max),"tr":(x_min,y_min)}


