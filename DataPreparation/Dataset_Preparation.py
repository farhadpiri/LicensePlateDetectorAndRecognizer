from Config import Addresses
import os
import glob
from DataStructures import ImageData

class PrepareDate:
    def __init__(self):
        pass

    def read_dataset1(self):
        address = os.path.join(Addresses.datasets_root_address,"datasets/01")
        images_address = os.path.join(address,"images/*.png")
        images_path = glob.glob(images_address)
        annotations_address = os.path.join(address, "annotations/*.xml")
        annotations_path = glob.glob(annotations_address)

        image_data = []
        for p in
        imageData = ImageData.ImageData()
        imageData.readImageData(image_path,annotation_path)



if(__name__=="__main__"):
    pass