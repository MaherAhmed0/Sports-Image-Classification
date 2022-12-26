import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import os
import imghdr
import random
import tflearn
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


def preprocessed_train_data(path):
    data = []
    converted_images = 0
    for category in CATEGORIES:
        data_path = os.path.join(path, category)
        label = CATEGORIES.index(category)
        print("loading", category, "images and labeling them")
        for img in tqdm(os.listdir(data_path)):
            img_array = cv2.imread(os.path.join(data_path, img))

            ### Preprocessing ###
            ###############################
            # convert png to jpg
            if imghdr.what(os.path.join(data_path, img)) == 'png':
                cv2.imwrite(os.path.join(data_path, img), img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                converted_images += 1

            # BGR to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # Resize
            img_array = cv2.resize(img_array, (100, 100))

            # Normalization
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
            ###############################
            data.append([img_array, label])
        print("Done for", category, "images.")
        print("------------------------------")
    print(converted_images, "images got converted from PNG to JPG")
    print("Total train images:", len(data))
    print("------------------------------")
    return data


def preprocessed_test_data(path):
    data = []
    image_name = []
    converted_images = 0
    print("loading test images")
    for img in os.listdir(path):
        image_name.append(img)
        test_img_array = cv2.imread(os.path.join(path, img))
        ### Preprocessing ###
        ###############################
        # convert png to jpg
        if imghdr.what(os.path.join(path, img)) == 'png':
            cv2.imwrite(os.path.join(path, img), test_img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            os.remove(os.path.join(path, img))
            converted_images += 1

        # BGR to RGB
        test_img_array = cv2.cvtColor(test_img_array, cv2.COLOR_BGR2RGB)
        # Resize
        test_img_array = cv2.resize(test_img_array, (100, 100))
        # Normalization
        test_img_array = (test_img_array - np.min(test_img_array)) / (np.max(test_img_array) - np.min(test_img_array))
        ###############################
        data.append(test_img_array)
    print("Done.")
    print(converted_images, "images got converted from PNG to JPG")
    print("Total test images:", len(data))
    print("------------------------------")
    return data, image_name


def test_model(model, test_data):
    c = 1
    labels = []
    print("Predictions:")
    for img in test_data:
        print("image:", c)
        prediction = model.predict([img])[0]
        print(f'Basketball: {prediction[0] * 100:.2f}%, Football: {prediction[1] * 100:.2f}%, '
              f'Rowing: {prediction[2] * 100:.2f}%, Swimming: {prediction[3] * 100:.2f}%, '
              f'Tennis: {prediction[4] * 100:.2f}%, Yoga:{prediction[5] * 100:.2f}%')

        labels.append(prediction.argmax())
        c += 1
    return labels


def augmentation(data: list):
    for entry in data:  # entry = [image, Tennis]
        entry: list
        label = CATEGORIES[entry[1]]
        img = entry[0]
        if COUNTS[label] < 480:  # Up Sampling 480
            Img_Aug = tflearn.ImageAugmentation()
            ### random Rotation ###
            Img_Aug.add_random_rotation()

            ### random Rotation with 90 deg ###
            r = random.randint(1, 3)
            Img_Aug = tf.image.rot90(img, r)
            Img_Aug = tf.compat.v1.Session().run(Img_Aug)
            data.append([Img_Aug, entry[1]])

            ### random flip lift & right ###
            Img_Aug = tf.image.flip_left_right(img)
            Img_Aug = tf.compat.v1.Session().run(Img_Aug)
            data.append([Img_Aug, entry[1]])

            COUNTS[label] += 2


def get_x_y_train(loaded_data):
    X_Train_ = []
    Y_Train_ = []
    for img_arr, img_label in loaded_data:
        X_Train_.append(img_arr)
        Y_Train_.append(img_label)

    return X_Train_, Y_Train_


def show_train_image(x, y, index):
    plt.imshow(x[index], cmap='gray')
    plt.xlabel(CATEGORIES[y[index]])
    plt.show()


def show_test_image(arr, index):
    plt.imshow(arr[index], cmap='gray')
    plt.show()


def save_data(file_name, data):
    pickle_file = open(file_name, "wb")
    pickle.dump(data, pickle_file)
    pickle_file.close()


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


### Preparing Training Data ###
Train_Data_Path = f"{current_dir}\\try"

Train_Data = preprocessed_train_data(Train_Data_Path)
augmentation(Train_Data)

X_Train_Data, Y_Train_Data = get_x_y_train(Train_Data)

save_data("X_Train", X_Train_Data)
del X_Train_Data
save_data("Y_Train", Y_Train_Data)
del Y_Train_Data

X_Train = retrieve_pickled_data("X_Train")
Y_Train = retrieve_pickled_data("Y_Train")
###############################
### Model ###
x_train, x_test, y_train, y_test = train_test_split(X_Train, Y_Train, train_size=0.8, shuffle=True)
Model = Models.inception_blocks(x_train, x_test, y_train, y_test)
###############################
### Preparing Testing Data ###
Test_Data_Path = f"{current_dir}\\small_test"
Test_Data, Test_images_Name = preprocessed_test_data(Test_Data_Path)
###############################
### Testing the model ###
Test_images_labels = test_model(Model, Test_Data)
###############################
### Generate Results File ###
print("------------------------------")
print("Images: ")
print(Test_images_Name)
print("------------------------------")
print("Classes: ")
print(Test_images_labels)
print("------------------------------")
Classified_Images = list(zip(Test_images_Name, Test_images_labels))
generate_csv(Classified_Images)
###############################
save_data("Test images", Test_Data)
# Test = retrieve_pickled_data("Test images")
