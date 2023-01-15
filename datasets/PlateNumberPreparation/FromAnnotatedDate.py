import os.path

import pandas as pd
import ntpath
import os
import cv2

def read_excel():
    data_store = "F:\edu\LicensePlateDetector\datasets\DataPreparation\image_store"
    df = pd.read_excel("data.xlsx")

    for idx, row in df.iterrows():
        if(idx < 0):
            continue
        path = row["image_path"]
        head, tail = ntpath.split(path)
        new_path = os.path.join(data_store,tail)
        img = cv2.imread(new_path)
        license_number = row["license_number"]