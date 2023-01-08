import os.path
import random
import shutil

import cv2
import pandas as pd
import ntpath

class PrepareForYOLO:
    def __init__(self):
        self.train_folder = os.path.join(os.getcwd(),"YoloDataset/train")
        if(not os.path.isdir(self.train_folder)):
            os.mkdir(self.train_folder)
        self.val_folder = os.path.join(os.getcwd(),"YoloDataset/val")
        if(not os.path.isdir(self.val_folder)):
            os.mkdir(self.val_folder)
        self.test_folder = os.path.join(os.getcwd(),"YoloDataset/test")
        if(not os.path.isdir(self.test_folder)):
            os.mkdir(self.test_folder)

    def copy_image(self,path):
        pass

    def create_txt(self):
        pass

    def create_yaml(self):
        pass

    def create_YOLO_dataset(self):
        df = self.read_dataframe("data.xlsx")

        for index, row in df.iterrows():
            path = row["image_path"]

            head, tail = ntpath.split(path)
            splited_tail = tail.split('.')
            name = splited_tail[0]
            format = splited_tail[-1]
            print(tail)
            src_img_path = os.path.join(os.getcwd(),"image_store",tail)

            r = random.randint(0,10)
            if(r < 8):
                train_image_dir = os.path.join(self.train_folder,"images")
                train_label_dir = os.path.join(self.train_folder,"labels")
                if(not os.path.isdir(train_image_dir)):
                    os.mkdir(train_image_dir)
                if (not os.path.isdir(train_label_dir)):
                    os.mkdir(train_label_dir)
                dst_img_path = os.path.join(train_image_dir,tail)
                dst_txt_path = os.path.join(train_label_dir,f"{name}.txt")
            elif(r < 9):
                val_image_dir = os.path.join(self.val_folder,"images")
                val_label_dir = os.path.join(self.val_folder,"labels")
                if(not os.path.isdir(val_image_dir)):
                    os.mkdir(val_image_dir)
                if (not os.path.isdir(val_label_dir)):
                    os.mkdir(val_label_dir)
                dst_img_path = os.path.join(val_image_dir,tail)
                dst_txt_path = os.path.join(val_label_dir,f"{name}.txt")
            else:
                test_image_dir = os.path.join(self.test_folder,"images")
                test_label_dir = os.path.join(self.test_folder,"labels")
                if(not os.path.isdir(test_image_dir)):
                    os.mkdir(test_image_dir)
                if (not os.path.isdir(test_label_dir)):
                    os.mkdir(test_label_dir)
                dst_img_path = os.path.join(test_image_dir,tail)
                dst_txt_path = os.path.join(test_label_dir,f"{name}.txt")


            # prepare_output(src_img_path)
            if(not os.path.isfile(dst_img_path)):
                shutil.copyfile(src_img_path, dst_img_path)

            img = cv2.imread(src_img_path)

            [H,W,C] = img.shape
            xmin = int(row["xmin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            ymin = int(row["ymin"])

            x_c = int(xmin + (xmax - xmin)/2)
            y_c = int(ymin + (ymax - ymin)/2)
            w = (xmax - xmin)/W
            h = (ymax - ymin)/H

            with open(dst_txt_path, "a+") as f:
                line = f"0 {x_c} {y_c} {w} {h}\n"
                f.write(line)



    def read_dataframe(self, path):
        return pd.read_excel(path)


if(__name__=="__main__"):
    prepare_dataset = PrepareForYOLO()
    prepare_dataset.create_YOLO_dataset()