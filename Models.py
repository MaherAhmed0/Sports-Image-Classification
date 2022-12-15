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
from tflearn.metrics import Accuracy


def model_1(x_train, x_test, y_train, y_test):
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

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(),
                            name='targets', to_one_hot=True, n_classes=6)
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    print("Start training...")
    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=5, show_metric=True,
              validation_set=({'input': x_test}, {'targets': y_test}))
    print("Finished...")
    return model


def model_2(x_train, x_test, y_train, y_test):
    conv_input = input_data(shape=[None, 50, 50, 3], name='input')

    conv1 = conv_2d(conv_input, 30, 3, activation='relu')
    pool1 = max_pool_2d(conv1, 2)

    conv2 = conv_2d(pool1, 30, 3, activation='relu')
    pool2 = max_pool_2d(conv2, 2)

    conv3 = conv_2d(pool2, 40, 3, activation='relu')
    pool3 = max_pool_2d(conv3, 2)

    conv4 = conv_2d(pool3, 40, 3, activation='relu')
    pool4 = max_pool_2d(conv4, 2)

    conv5 = conv_2d(pool4, 40, 3, activation='relu')
    pool5 = max_pool_2d(conv5, 2)

    conv6 = conv_2d(pool5, 30, 3, activation='relu')
    pool6 = max_pool_2d(conv6, 2)

    cnn_layers = fully_connected(pool6, 6, activation='softmax')
    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(),
                            name='targets', to_one_hot=True, n_classes=6)

    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    print("Start training...")
    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=5, show_metric=True,
              validation_set=({'input': x_test}, {'targets': y_test}))
    print("Finished...")
    return model


def model_3(x_train, x_test, y_train, y_test):
    conv_input = input_data(shape=[None, 50, 50, 3], name='input')

    conv1 = conv_2d(conv_input, 10, 3, activation='relu')
    conv2 = conv_2d(conv1, 10, 3, activation='relu')

    pool1 = max_pool_2d(conv2, 2)

    conv3 = conv_2d(pool1, 10, 3, activation='relu')
    conv4 = conv_2d(conv3, 10, 3, activation='relu')

    pool2 = max_pool_2d(conv4, 2)

    conv5 = conv_2d(pool2, 10, 3, activation='relu')
    conv6 = conv_2d(conv5, 10, 3, activation='relu')

    pool3 = max_pool_2d(conv6, 2)

    fully_layer1 = fully_connected(pool3, 512, activation='relu')
    fully_layer3 = fully_connected(fully_layer1, 1024, activation='relu')

    cnn_layers = fully_connected(fully_layer3, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(),
                            name='targets', to_one_hot=True, n_classes=6)
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    print("Start training...")
    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=5, show_metric=True,
              validation_set=({'input': x_test}, {'targets': y_test}))
    print("Finished...")
    return model


def model_4(x_train, x_test, y_train, y_test):
    conv_input = input_data(shape=[None, 50, 50, 3], name='input')

    conv1 = conv_2d(conv_input, 96, 11, activation='relu')
    conv2 = conv_2d(conv1, 256, 5, activation='relu')

    pool1 = max_pool_2d(conv2, 3)

    conv3 = conv_2d(pool1, 384, 3, activation='relu')

    pool2 = max_pool_2d(conv3, 3)

    conv4 = conv_2d(pool2, 384, 3, activation='relu')
    conv5 = conv_2d(conv4, 256, 3, activation='relu')

    pool3 = max_pool_2d(conv5, 3)

    fully_layer1 = fully_connected(pool3, 4096, activation='relu')
    fully_layer2 = fully_connected(fully_layer1, 4096, activation='relu')
    fully_layer3 = fully_connected(fully_layer2, 1000, activation='relu')

    cnn_layers = fully_connected(fully_layer3, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(),
                            name='targets', to_one_hot=True, n_classes=6)
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    print("Start training...")
    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=5, show_metric=True,
              validation_set=({'input': x_test}, {'targets': y_test}))
    print("Finished...")
    return model
