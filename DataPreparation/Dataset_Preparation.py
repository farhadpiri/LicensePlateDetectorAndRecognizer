from Config import Addresses
import os
import glob

class PrepareDate:
    def __init__(self):
        pass

    def read_dataset1(self):
        address = os.path.join(Addresses.datasets_root_address,"datasets/01")
        images_address = os.path.join(address,"images/*.png")
        images_path = glob.glob(images_address)
        annotations_address = os.path.join(address, "annotations/*.xml")
        annotations_path = glob.glob(annotations_address)



if(__name__=="__main__"):
    pass