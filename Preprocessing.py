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
from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d, global_avg_pool
from tflearn.layers.core import input_data, dropout, fully_connected,flatten
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn.layers.merge_ops import merge

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

Train_Data_Path = "drive/MyDrive/NN_project/N_Train"
CATEGORIES = ["Basketball", "Football", "Rowing", "Swimming", "Tennis", "Yoga"]
Train_Data = []

preprocessed_train_data(Train_Data, Train_Data_Path)

random.shuffle(Train_Data)

X_Train_Data = []
Y_Train_Data = []

for img_data, img_label in Train_Data:
    X_Train_Data.append(img_data)
    Y_Train_Data.append(img_label)

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

def Inception_a (layer,kernal_size):
  #branch1
  branch1_1 = conv_2d(layer, 64,1,activation='relu')
  branch1_2 = conv_2d(branch1_1, 96,3,activation='relu')
  branch1_3 = conv_2d(branch1_2, 96,3,activation='relu')
  #branch2
  branch2_1 = conv_2d(layer, 48,1,activation='relu')
  branch2_2 = conv_2d(branch2_1, 64,3,activation='relu')
  #branch3
  branch3_1 = avg_pool_2d(layer,3,strides=1)
  branch3_2 = conv_2d(branch3_1, 48, kernal_size,1,activation='relu')
  #branch4
  branch4 = conv_2d(layer, 64,1,activation='relu')

  result = merge([branch1_3,branch2_2,branch3_2,branch4], 'concat',axis =3)

  return result

def Reduction_a(layer):
    #branch1
    branch1 =  conv_2d(layer,64,1,activation='relu')
    branch1 =  conv_2d(branch1,96,3,activation='relu')
    branch1 =  conv_2d(branch1,96,3,activation='relu',strides = 2)
    #branch2
    branch2 = conv_2d(layer,384,3,activation='relu',strides = 2)
    #branch3
    branch3 = max_pool_2d(layer,3, strides = 2 )
    
    result = merge([branch1,branch2,branch3],'concat',axis =3)
    
    return result
def Inception_b(layer, kernal_size):
    #branch1
    branch1 =  conv_2d(layer,kernal_size,1,activation='relu')
    branch1 =  conv_2d(branch1,kernal_size,(7,1),activation='relu')
    branch1 =  conv_2d(branch1,kernal_size,(1,7),activation='relu')
    branch1 =  conv_2d(branch1,kernal_size,(7,1),activation='relu')
    branch1 =  conv_2d(branch1,192,(1,7),activation='relu')
    #branch2
    branch2 = conv_2d(layer,kernal_size,(1,1),activation='relu')
    branch2 = conv_2d(branch2,kernal_size,(1,7),activation='relu')
    branch2 = conv_2d(branch2,192,(7,1),activation='relu')
    #branch3
    branch3 = avg_pool_2d(layer,3,strides=1)
    branch3 = conv_2d(branch3,192,(1,1),activation='relu')
    #branch4
    branch4 = conv_2d(layer,192,(1,1),activation='relu')
    
    result = merge([branch1,branch2,branch3,branch4],'concat',axis =3)
    
    return result   

def auxiliary(Layer):
    x = avg_pool_2d(Layer,5 , strides=3)
    x = conv_2d(x,128,1,activation='relu')
    x = flatten(x)
    x = fully_connected(x,768, activation='relu')
    x = dropout(x,0.5)
    x = fully_connected(x,1000, activation='softmax')
    return x


def Reduction_b(layer):
    #branch1
    branch1_1 =  conv_2d(layer,192,(1,1),activation='relu')
    branch1_2 =  conv_2d(branch1_1,192,(1,7),activation='relu')
    branch1_3 =  conv_2d(branch1_2,192,(7,1),activation='relu')
    branch1_4 =  conv_2d(branch1_3,192, (3,3), strides=(2,2) , padding = 'valid',activation='relu')
    #branch2
    branch2_1 = conv_2d(layer,192,(1,1),activation='relu')
    branch2_2 = conv_2d(branch2_1,320,(3,3), strides=(2,2) , padding = 'valid',activation='relu')
    #branch3
    branch3 = max_pool_2d(layer,3,strides = 2)

    result =  merge([branch1_4,branch2_2,branch3],'concat',axis=3)
    
    return result

def Inception_c(layer):
    #branch1
    branch1 =  conv_2d(layer,448,1,activation='relu')
    branch1 =  conv_2d(branch1,384,(3,3),activation='relu')
    branch1_1 =  conv_2d(branch1,384,(1,3),activation='relu')
    branch1_2 =  conv_2d(branch1,384, (3,1),activation='relu')
    branch1 = merge([branch1_1,branch1_2],'concat',axis =0)
    #branch2
    branch2 = conv_2d(layer,384,(1,1),activation='relu')
    branch2_1 = conv_2d(branch2,384,(1,3),activation='relu')
    branch2_2 = conv_2d(branch2,384,(3,1),activation='relu')
    branch2 = merge([branch2_1,branch2_2],'concat',axis =0)
    #branch3
    branch3 = avg_pool_2d(layer,3,strides=1)
    branch3 = conv_2d(branch3,192,(1,1),activation='relu')
    #branch4
    branch4 = conv_2d(layer,320,(1,1),activation='relu')
    
    result = merge([branch1,branch2,branch3,branch4],'concat',axis =3)
    
    return result

def inception(x_train, x_test, y_train, y_test):
    conv_input = input_data(shape=[None, 100, 100, 3], name='input')

    conv1 = conv_2d(conv_input, 32, 3, strides = 2,activation='relu')
    conv2 = conv_2d(conv1, 32, 3,strides = 1 ,activation='relu')
    conv3 = conv_2d(conv2, 64, 3,strides = 1 , activation='relu',padding='same')

    pool1 = max_pool_2d(conv3, 3,strides = 2)

    conv4 = conv_2d(pool1, 80, 1, activation='relu')
    conv5 = conv_2d(conv3, 192, 3, activation='relu')

    pool2 = max_pool_2d(conv5, 3,strides = 2)

    inception_a_ = Inception_a(pool2,32)
    inception_a_ = Inception_a(inception_a_,64)
    inception_a_ = Inception_a(inception_a_,64)
    
    Reduction_a_ = Reduction_a(inception_a_)
    
    inception_b_ = Inception_b(Reduction_a_,128)
    inception_b_ = Inception_b(inception_b_,160)
    inception_b_ = Inception_b(inception_b_,160)
    inception_b_ = Inception_b(inception_b_,192)

    aux = auxiliary(inception_b_)

    #Reduction_b_ =  Reduction_b(inception_b_)

    inception_c_ = Inception_c(inception_b_) 
    inception_c_ = Inception_c(inception_c_) 

    X =global_avg_pool(inception_c_)

    fully_layer1 = fully_connected(X, 2048, activation='relu')

    S = dropout(fully_layer1,0.2)

    fully_layer2 = fully_connected(S, 1000, activation='relu')
    
    

    cnn_layers = fully_connected(fully_layer2, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(),name='targets', to_one_hot=True, n_classes=6)
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    print("Start training...")

    if (os.path.exists('model.meta')):
        model.load('./model')
    else:
        model.fit({'input': x_train}, {'targets': [y_train,aux]}, n_epoch=1, show_metric=True,
              validation_set=({'input': x_test}, {'targets': y_test}))
       # model.save('model')

    print("Finished...")
    return model

Model_ = inception(x_train, x_test, y_train, y_test)


Test_Data_Path = "drive/MyDrive/NN_project/N_Test"
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

