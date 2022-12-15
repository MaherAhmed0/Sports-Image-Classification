import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import os
import imghdr
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from PIL import Image
import re
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import Models


def preprocessed_train_data(data, path):
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
            # img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

            # Normalization
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
            ###############################
            data.append([img_array, label])
        print("Done for", category, "images.")
        print("------------------------------")
        # break
    print(converted_images, "images got converted from PNG to JPG")
    print("Total train images:", len(data))
    print("------------------------------")


def preprocessed_test_data(data, path, image_name):
    converted_images = 0
    print("loading test images")
    for img in os.listdir(path):
        image_name.append(img)
        # print(img)
        test_img_array = cv2.imread(os.path.join(path, img))
        if imghdr.what(os.path.join(path, img)) == 'png':
            cv2.imwrite(os.path.join(path, img), test_img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            os.remove(os.path.join(path, img))
            converted_images += 1
        test_img_array = cv2.cvtColor(test_img_array, cv2.COLOR_BGR2RGB)
        test_img_array = cv2.resize(test_img_array, (50, 50))
        data.append(test_img_array)
    print("Done.")
    print(converted_images, "images got converted from PNG to JPG")
    print("Total test images:", len(data))
    print("------------------------------")


def test_model(model, test_data, labels):
    c = 1
    for img in test_data:
        print("image:", c)
        prediction = model.predict([img])[0]
        print(f'Basketball: {prediction[0]*100:.2f}%, Football: {prediction[1]*100:.2f}%, Rowing: {prediction[2]*100:.2f}%, '
              f'Swimming: {prediction[3]*100:.2f}%, Tennis: {prediction[4]*100:.2f}%, Yoga:{prediction[5]*100:.2f}%')
        labels.append(prediction.argmax())
        c += 1


def show_train_image(x, y, index):
    plt.imshow(x[index], cmap='gray')
    plt.xlabel(CATEGORIES[y[index]])
    plt.show()


def show_test_image(arr, index):
    plt.imshow(arr[index], cmap='gray')
    plt.show()


def retrieve_pickled_data(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def generate_csv(classified_images):
    fields = ['image_name', 'label']
    with open('Results', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(classified_images)


Train_Data_Path = "D:\\Courses\\Neural Network and Deep Learning\\Project\\Sports Image Classification\\try"
CATEGORIES = ["Basketball", "Football", "Rowing", "Swimming", "Tennis", "Yoga"]
Train_Data = []

preprocessed_train_data(Train_Data, Train_Data_Path)
random.shuffle(Train_Data)
X_Train_Data = []
Y_Train_Data = []

for img_data, img_label in Train_Data:
    X_Train_Data.append(img_data)
    Y_Train_Data.append(img_label)

# show_train_image(X_Train_Data, Y_Train_Data, 0)

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

x_train, x_test, y_train, y_test = train_test_split(X_Train, Y_Train, train_size=0.8)

# model
Model_ = Models.model_1(x_train, x_test, y_train, y_test)

Test_Data_Path = "D:\\Courses\\Neural Network and Deep Learning\\Project\\Sports Image Classification\\test"
Test_Data = []
Test_images_Name = []
Test_images_labels = []
# Test = retrieve_pickled_data("Test images")

preprocessed_test_data(Test_Data, Test_Data_Path, Test_images_Name)
# show_test_image(Test_Data, 0)

test_model(Model_, Test_Data, Test_images_labels)
print("------------------------------")
print(Test_images_Name)
print("------------------------------")
print(Test_images_labels)
print("------------------------------")
Classified_Images = list(zip(Test_images_Name, Test_images_labels))
# print(Classified_Images)
generate_csv(Classified_Images)

pickle_out = open("Test images", "wb")
pickle.dump(Test_Data, pickle_out)
pickle_out.close()
del Test_Data
