import os.path
from Config.Addresses import intigrated_dataset_address,datasets_root_address
from glob import glob
import shutil
import ntpath

class Prepare_Data:
    def __init__(self):
        pass

    def prepare_1(self):
        pass


    def move_data(self,src_address,dst_address):
        image_paths = glob(f"{src_address}/*/*.jpg", recursive=True)
        for path in image_paths:
            head, tail = ntpath.split(path)
            dst_path = os.path.join(dst_address,tail)
            shutil.copyfile(path, dst_path)

        xml_paths = glob(f"{src_address}/*/*.xml", recursive=True)
        for path in xml_paths:
            head, tail = ntpath.split(path)
            dst_path = os.path.join(dst_address,tail)
            shutil.copyfile(path, dst_path)


if(__name__=="__main__"):
    src_address = os.path.join(os.getcwd(),"03","State-wise_OLX")
    dst_address = os.path.join(os.getcwd(),"integrated")
