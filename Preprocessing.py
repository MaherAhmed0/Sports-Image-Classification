import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import imghdr
import random
from tqdm import tqdm
import pickle
from PIL import Image
import re
import tensorflow as tf


def create_preprocessed_data(data, path):
    converted_images = 0
    for category in CATEGORIES:
        data_path = os.path.join(path, category)
        label = CATEGORIES.index(category)
        print("loading", category, "images and labeling them")
        # c = 0
        for img in tqdm(os.listdir(data_path)):
            img_array = cv2.imread(os.path.join(data_path, img))

            ### Preprocessing ###
            ###############################
            # convert png to jpg
            if imghdr.what(os.path.join(data_path, img)) == 'png':
                cv2.imwrite(os.path.join(data_path, img), img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                os.remove(os.path.join(data_path, img))
                converted_images += 1

            # BGR to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # Resize
            img_array = cv2.resize(img_array, (50, 50))
            # print("Shape: ", img_array.shape)

            # Smoothing
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

            # Normalization
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
            ###############################
            data.append([img_array, label])
            # if c == 2:
            #     break
            # c += 1
            # break
        print("Done for", category, "images.")
        print("------------------------------")
        # break
    print(converted_images, "images got converted from PNG to JPG")
    print("------------------------------")


def show_image(x, y, index):
    plt.imshow(x[index], cmap='gray')
    plt.xlabel(CATEGORIES[y[index]])
    plt.show()


def retrieve_pickled_data(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


Data_path = "D:\\Courses\\Neural Network and Deep Learning\\Project\\Sports-Image-Classification\\try"
CATEGORIES = ["Basketball", "Football", "Rowing", "Swimming", "Tennis", "Yoga"]
Train_Data = []

create_preprocessed_data(Train_Data, Data_path)
print("Total images:", len(Train_Data))
print("------------------------------")

random.shuffle(Train_Data)
X_Train_Data = []
Y_Train_Data = []

for img_data, img_label in Train_Data:
    X_Train_Data.append(img_data)
    Y_Train_Data.append(img_label)

show_image(X_Train_Data, Y_Train_Data, 0)
show_image(X_Train_Data, Y_Train_Data, 5)
show_image(X_Train_Data, Y_Train_Data, 10)
show_image(X_Train_Data, Y_Train_Data, 15)
show_image(X_Train_Data, Y_Train_Data, 20)
show_image(X_Train_Data, Y_Train_Data, 25)

# print(Train_Data)
# print(X_Train_Data)
# print(Y_Train_Data)

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

