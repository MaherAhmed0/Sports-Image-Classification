import tensorflow as tf
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
import Models

current_dir = os.getcwd()

basketball = 'Basketball'
football = 'Football'
rowing = 'Rowing'
swimming = 'Swimming'
tennis = 'Tennis'
yoga = 'Yoga'

CATEGORIES = [basketball, football, rowing, swimming, tennis, yoga]
COUNTS = {
    basketball: 196,
    football: 400,
    rowing: 202,
    swimming: 240,
    tennis: 185,
    yoga: 458,
}


def preprocessed_train_data(data, path):
    converted_images = 0
    for category in CATEGORIES:
        data_path = os.path.join(path, category)
        label = CATEGORIES.index(category)
        print("loading", category, "images and labeling them")
        # c = 0
        for img in tqdm(os.listdir(data_path)):
            img_array = cv2.imread(os.path.join(data_path, img))

            # Preprocessing #
            ###############################
            # convert png to jpg
            if imghdr.what(os.path.join(data_path, img)) == 'png':
                cv2.imwrite(os.path.join(data_path, img), img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # os.remove(os.path.join(data_path, img))
                converted_images += 1

            # BGR to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # Resize
            img_array = cv2.resize(img_array, (100, 100))
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
        test_img_array = cv2.resize(test_img_array, (100, 100))
        # Normalization
        test_img_array = (test_img_array - np.min(test_img_array)) / (np.max(test_img_array) - np.min(test_img_array))
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
        print(
            f'Basketball: {prediction[0] * 100:.2f}%, Football: {prediction[1] * 100:.2f}%, '
            f'Rowing: {prediction[2] * 100:.2f}%, Swimming: {prediction[3] * 100:.2f}%, '
            f'Tennis: {prediction[4] * 100:.2f}%, Yoga:{prediction[5] * 100:.2f}%'
        )
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


def augmentation(data: list):
    for entry in data:  # x = [image, Tennis]
        entry: list
        label = CATEGORIES[entry[1]]
        img = entry[0]
        if COUNTS[label] < 480:
            r = random.randint(1, 3)
            aug = tf.image.rot90(img, r)
            aug = tf.compat.v1.Session().run(aug)
            data.append([aug, entry[1]])

            aug = tf.image.random_brightness(img, 0.5)
            aug = tf.compat.v1.Session().run(aug)
            data.append([aug, entry[1]])

            COUNTS[label] += 2


Train_Data_Path = f"{current_dir}\\try"
Train_Data = []

preprocessed_train_data(Train_Data, Train_Data_Path)
augmentation(Train_Data)
# random.shuffle(Train_Data)

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

x_train, x_test, y_train, y_test = train_test_split(X_Train, Y_Train, train_size=0.8, shuffle=True)

# model
Model = Models.inception_v3(x_train, x_test, y_train, y_test)
Model.save('inception_v3.tfl')

Test_Data_Path = f"{current_dir}\\small_test"
Test_Data = []
Test_images_Name = []
Test_images_labels = []

preprocessed_test_data(Test_Data, Test_Data_Path, Test_images_Name)

test_model(Model, Test_Data, Test_images_labels)
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
# Test = retrieve_pickled_data("Test images")
