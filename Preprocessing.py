import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os
import imghdr
import random
from tqdm import tqdm
import pickle
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def create_preprocessed_data(data, path):
    converted_images = 0
    for category in CATEGORIES:
        data_path = os.path.join(path, category)
        print('path/name',data_path)
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


Data_path = "D:\\Sports-Image-Classification\\try"
CATEGORIES = ["Basketball", "Football", "Rowing", "Swimming", "Tennis", "Yoga"]
Train_Data = []
Test_Data = []

create_preprocessed_data(Train_Data, Data_path)
print("Total images:", len(Train_Data))
print("------------------------------")

random.shuffle(Train_Data)
X_Train_Data = []
Y_Train_Data = []

X_Test_Data = []
Y_Test_Data = []

for img_data, img_label in Train_Data:
    X_Train_Data.append(img_data)
    Y_Train_Data.append(img_label)
    
# for img_data, img_label in Train_Data:
#     X_Test_Data.append(img_data)
#     Y_Test_Data.append(img_label)
    

#show_image(X_Train_Data, Y_Train_Data, 0)
# show_image(X_Train_Data, Y_Train_Data, 5)
# show_image(X_Train_Data, Y_Train_Data, 10)
# show_image(X_Train_Data, Y_Train_Data, 15)
# show_image(X_Train_Data, Y_Train_Data, 20)
# show_image(X_Train_Data, Y_Train_Data, 25)

# print(Train_Data)
# print(X_Train_Data)
# print(Y_Train_Data)

# pickle_out = open("X_Train", "wb")
# pickle.dump(X_Train_Data, pickle_out)
# pickle_out.close()
# del X_Train_Data

# pickle_out = open("Y_Train", "wb")
# pickle.dump(Y_Train_Data, pickle_out)
# pickle_out.close()
# del Y_Train_Data

# pickle_out = open("X_Test", "wb")
# pickle.dump(X_Test_Data, pickle_out)
# pickle_out.close()
# del X_Test_Data

# pickle_out = open("Y_Test", "wb")
# pickle.dump(Y_Test_Data, pickle_out)
# pickle_out.close()
# del Y_Test_Data


# X_Train = retrieve_pickled_data("X_Train")
# Y_Train = retrieve_pickled_data("Y_Train")

# X_Test = retrieve_pickled_data("X_Test")
# Y_Test = retrieve_pickled_data("Y_Test")

# X_Train = np.array([i[0] for i in Train_Data]).reshape(-1, 50, 50, 1)
# Y_Train = [i[1] for i in Train_Data]

# X_Test = np.array([i[0] for i in Test_Data]).reshape(-1, 50, 50, 1)
# Y_Test = [i[1] for i in Test_Data]

x_train,x_test,y_train,y_test = train_test_split(X_Train_Data,Y_Train_Data,train_size = 0.8)

#model
conv_input = input_data(shape=[None, 50, 50, 3], name='input')
conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)

fully_layer = fully_connected(pool5, 1024, activation='relu')


cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets' , to_one_hot = True , n_classes = 6)
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
print (len(x_train))
print (len(y_train))
print (len(x_test))
print (len(y_test))

model.fit({'input': x_train}, {'targets': y_train}, n_epoch=100,
        validation_set=({'input': x_test}, {'targets': y_test}))
print("wwwwwwwwwwwwwwwwwwwwwwwwwwww")

    
img = cv2.imread('D:\\Sports-Image-Classification\\try\\Basketball\\Basketball_1.jpg',0)
test_img = cv2.resize(img, (50, 50))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
#test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
prediction = model.predict([test_img])[0]
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.imshow(test_img,cmap='gray')
print(f"basketball: {prediction[0]}, football: {prediction[1]}, Rowing: {prediction[2]}, swimming: {prediction[3]},tennis: {prediction[4]},yoga:{prediction[5]}")
plt.show()
#s
# print(X_Train)ss
# print(Y_Train)

