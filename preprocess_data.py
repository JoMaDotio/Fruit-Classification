import numpy as np
import pandas as pnd
import cv2 as cv
import glob as gb
import os

from data_analysis import get_size

wanted_path = "./normaliced_dataset/"

source_path = "./fruit_dataset/"

size = get_size()
if(not os.path.exists(wanted_path)):
    os.makedirs(wanted_path)
for folder in os.listdir(source_path):
    if(not os.path.exists(wanted_path + folder)):
        os.makedirs(wanted_path + folder.lower())
    if(source_path+folder == "./fruit_dataset/predict"):
        for file in os.listdir(source_path+folder):
            img = cv.imread(source_path+folder+"/"+file)
            resize_img = cv.resize(img, (size, size))
            status = cv.imwrite(
                wanted_path+folder.lower()+"/"+file, resize_img)
            print(f"Image saved status: {status}")
        continue
    for subFolder in os.listdir(source_path+folder):
        if (not os.path.exists(wanted_path + folder + "/" + subFolder)):
            os.makedirs(wanted_path + folder + "/" + subFolder.lower())
        for file in os.listdir(source_path+folder+"/"+subFolder):
            img = cv.imread(f"{source_path}{folder}/{subFolder}/{file}")
            if img is None:
                continue
            resize_img = cv.resize(img, (size, size))
            status = cv.imwrite(
                f"{wanted_path}{folder}/{subFolder.lower()}/{file}", resize_img)
            print(f"Image status: {status}")
