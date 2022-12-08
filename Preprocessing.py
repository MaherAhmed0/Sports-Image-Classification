import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from tqdm import tqdm
import pickle
import re


def create_data(data, path):
    for category in CATEGORIES:
        data_path = os.path.join(path, category)
        label = CATEGORIES.index(category)
        print("loading", category, "images and labeling them")
        for img in tqdm(os.listdir(data_path)):
            img_array = cv2.imread(os.path.join(data_path, img), cv2.IMREAD_GRAYSCALE)
            data.append([img_array, label])
        print("Done for", category, "images.")
        print("------------------------------")


def retrieve_pickled_data(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


Data_path = "D:\\Courses\\Neural Network and Deep Learning\\Project\\Sports image classification\\try"
CATEGORIES = ["Basketball", "Football", "Rowing", "Swimming", "Tennis", "Yoga"]
Train_Data = []
create_data(Train_Data, Data_path)
print("Total images:", len(Train_Data))
random.shuffle(Train_Data)

X_Train_Data = []
Y_Train_Data = []

for img_data, img_label in Train_Data:
    X_Train_Data.append(img_data)
    Y_Train_Data.append(img_label)

# print(Train_Data)
# print(X_Train)
# print(Y_Train)

pickle_out = open("X_Train", "wb")
pickle.dump(X_Train_Data, pickle_out)
pickle_out.close()
del X_Train_Data

pickle_out = open("Y_Train", "wb")
pickle.dump(Y_Train_Data, pickle_out)
pickle_out.close()
del Y_Train_Data

X_Train = retrieve_pickled_data("X_Train")
Y_Train = retrieve_pickled_data("Y_Train")

print(X_Train)
print(Y_Train)
