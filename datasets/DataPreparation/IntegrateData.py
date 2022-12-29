import glob
import os.path
import shutil
import ntpath
import pandas as pd
from bs4 import BeautifulSoup
from Config import Addresses

class DataIntegration:
    def __init__(self):
        self.DataFrame = None

    def create_df(self):
        self.DataFrame = pd.DataFrame({"image_path","plate_exist","xmin","xmax","xmin","ymax","license_number"})

    def prepare_1(self,dst_dir):
        images_dir = os.path.join(Addresses.datasets_root_address+"/01/images/*.png")
        images_paths = glob.glob(images_dir)
        for img_path in images_paths:
            # copy image
            head, tail = ntpath.split(img_path)
            name,format = tail.split('.')
            dst_path = os.path.join(dst_dir,tail)
            if(not os.path.isfile(dst_path)):
                shutil.copyfile(img_path, dst_path)

            # save annotation
            xml_path = img_path.replace("images","annotations").replace(".png",".xml")
            [plate_exist,xmin,ymin,xmax,ymax,license_number] = self.read_xml_1(xml_path)
            new_row = {"image_path":dst_path,"plate_exist":plate_exist,"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"license_number":license_number}
            self.DataFrame = self.DataFrame.append(new_row,ignore_index=True)


    def read_xml_1(self, xml_path):
        with open(xml_path,'r') as xml_file:
            file = xml_file.read()

        soup = BeautifulSoup(file,'xml')
        xmin = soup.findAll('xmin')[0].text
        ymin = soup.findAll('ymin')[0].text
        xmax = soup.findAll('xmax')[0].text
        ymax = soup.findAll('ymax')[0].text
        number = soup.findAll("name")[0].text
        if(number in ["licnse","num_plate"]):
            number = "Unspecified"
        return ["1",xmin,ymin,xmax,ymax,number]



    def prepare_2(self,dst_dir):
        images_dir = os.path.join(Addresses.datasets_root_address + "/02/images/*.jpeg")
        images_paths = glob.glob(images_dir)
        for img_path in images_paths:
            # copy image
            head, tail = ntpath.split(img_path)
            name, format = tail.split('.')
            dst_path = os.path.join(dst_dir, tail)
            if (not os.path.isfile(dst_path)):
                shutil.copyfile(img_path, dst_path)

            # save annotation
            xml_path = img_path.replace(".jpeg", ".xml")
            [plate_exist, xmin, ymin, xmax, ymax, license_number] = self.read_xml_1(xml_path)
            new_row = {"image_path": dst_path, "plate_exist": plate_exist, "xmin": xmin, "xmax": xmax, "ymin": ymin,
                       "ymax": ymax, "license_number": license_number}
            self.DataFrame = self.DataFrame.append(new_row, ignore_index=True)

    def prepare_3(self,dst_dir):
        images_dir_jpeg = os.path.join(Addresses.datasets_root_address + "/03/*/*.jpeg")
        images_dir_jpg = os.path.join(Addresses.datasets_root_address + "/03/*/*.jpg")
        images_paths = glob.glob(images_dir_jpeg) + glob.glob(images_dir_jpg)
        for img_path in images_paths:
            # copy image
            head, tail = ntpath.split(img_path)
            splited_tail = tail.split('.')
            name = splited_tail[0]
            format = splited_tail[-1]
            dst_path = os.path.join(dst_dir, f"{name}.{format}")
            if (not os.path.isfile(dst_path)):
                shutil.copyfile(img_path, dst_path)

            # save annotation
            if(format == "jpeg"):
                xml_path = img_path.replace(".jpeg", ".xml")
            else:
                xml_path = img_path.replace(".jpg", ".xml")
            [plate_exist, xmin, ymin, xmax, ymax, license_number] = self.read_xml_1(xml_path)
            new_row = {"image_path": dst_path, "plate_exist": plate_exist, "xmin": xmin, "xmax": xmax, "ymin": ymin,
                       "ymax": ymax, "license_number": license_number}
            self.DataFrame = self.DataFrame.append(new_row, ignore_index=True)

    def prepare_4(self):
        pass

    def prepare_5(self):
        pass

    def prepare_6(self):
        pass

    def prepare_7(self):
        images_dir_jpg = os.path.join(Addresses.datasets_root_address + "/07/*/images/*.jpg")
        images_paths = glob.glob(images_dir_jpg)
        for img_path in images_paths:
            # copy image
            head, tail = ntpath.split(img_path)
            splited_tail = tail.split('.')
            name = splited_tail[0]
            format = splited_tail[-1]
            dst_path = os.path.join(dst_dir, f"{name}.{format}")
            if (not os.path.isfile(dst_path)):
                shutil.copyfile(img_path, dst_path)

            # save annotation
            xml_path = img_path.replace(".jpg", ".xml")

            [plate_exist, xmin, ymin, xmax, ymax, license_number] = self.read_xml_1(xml_path)
            new_row = {"image_path": dst_path, "plate_exist": plate_exist, "xmin": xmin, "xmax": xmax, "ymin": ymin,
                       "ymax": ymax, "license_number": license_number}
            self.DataFrame = self.DataFrame.append(new_row, ignore_index=True)


if(__name__=="__main__"):
    integrate_data = DataIntegration()
    dst_dir = os.path.join(Addresses.datasets_root_address+"/image_store/")
    integrate_data.create_df()
    integrate_data.prepare_7(dst_dir)


