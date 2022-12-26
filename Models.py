import os
import tflearn
from tflearn import merge
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy


def inception_blocks(x_train, x_test, y_train, y_test):
    conv_input = input_data(shape=[None, 100, 100, 3], name='input')

    ### Stem Block ###
    conv_1 = conv_2d(conv_input, 32, 3, activation='relu', strides=2)
    conv_2 = conv_2d(conv_1, 32, 3, activation='relu', strides=1)
    conv_3 = conv_2d(conv_2, 64, 3, activation='relu', strides=1)

    pool_1 = max_pool_2d(conv_3, 3, strides=2)

    conv_4 = conv_2d(pool_1, 80, 1, activation='relu', strides=1)
    conv_5 = conv_2d(conv_4, 192, 3, activation='relu', strides=1)

    pool_2 = max_pool_2d(conv_5, 3, strides=2)
    ########################################
    ### Inception_A Block ###
    B1_conv_1 = conv_2d(pool_2, 64, 1, activation='relu')
    B1_conv_2 = conv_2d(B1_conv_1, 96, 3, activation='relu')
    B1_conv_3 = conv_2d(B1_conv_2, 96, 3, activation='relu')

    B2_conv_1 = conv_2d(pool_2, 48, 1, activation='relu')
    B2_conv_2 = conv_2d(B2_conv_1, 64, 3, activation='relu')

    B4_conv = conv_2d(pool_2, 64, 1, activation='relu')

    merge_1 = merge([B1_conv_3, B2_conv_2, B4_conv], 'concat', axis=3)
    ########################################
    ### Reduction Block ###
    Reduction_B1_conv_1 = conv_2d(merge_1, 64, 1, activation='relu')
    Reduction_B1_conv_2 = conv_2d(Reduction_B1_conv_1, 96, 1, activation='relu')
    Reduction_B1_conv_3 = conv_2d(Reduction_B1_conv_2, 96, 1, activation='relu', strides=2)

    Reduction_B2_conv = conv_2d(merge_1, 96, 1, activation='relu', strides=2)

    Reduction_B3_pool = max_pool_2d(merge_1, 3, strides=2)

    merge_2 = merge([Reduction_B1_conv_3, Reduction_B2_conv, Reduction_B3_pool], 'concat', axis=3)
    ########################################
    ### Fully Connected ###
    fully_layer = fully_connected(merge_2, 64, activation='relu')
    drop = dropout(fully_layer, 0.5)

    cnn_layers = fully_connected(drop, 6, activation='softmax')
    ########################################
    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(), name='targets', to_one_hot=True, n_classes=6)

    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3, best_checkpoint_path='BestAcc__')

    if os.path.exists('inception_v3.tfl.meta'):
        model.load('./inception_v3.tfl')
    else:
        print("Start training...")
        model.fit({'input': x_train}, {'targets': y_train}, n_epoch=50, show_metric=True, batch_size=32,
                  validation_set=({'input': x_test}, {'targets': y_test}))
        print("Finished...")
        model.save('inception_v3.tfl')

    return model


def model_6(x_train, x_test, y_train, y_test):
    conv_input = input_data(shape=[None, 100, 100, 3], name='input')

    conv_1 = conv_2d(conv_input, 48, 3, activation='relu')

    pool_1 = max_pool_2d(conv_1, 2, strides=2)

    conv_2 = conv_2d(pool_1, 48, 3, activation='relu', strides=2)

    pool_2 = max_pool_2d(conv_2, 2, strides=2)

    conv_3 = conv_2d(pool_2, 32, 3, activation='relu', strides=2)

    pool_3 = max_pool_2d(conv_3, 2, strides=2)

    flat = flatten(pool_3)

    fully_layer_1 = fully_connected(flat, 128, activation='relu')
    D1 = dropout(fully_layer_1, 0.5)

    fully_layer_2 = fully_connected(D1, 64, activation='relu')
    D2 = dropout(fully_layer_2, 0.5)

    cnn_layers = fully_connected(D2, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(), name='targets', to_one_hot=True, n_classes=6)

    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3, best_checkpoint_path='Best_Acc_')

    if os.path.exists('Best_Acc_.tfl8897.meta'):
        model.load('./Best_Acc_.tfl8897')
    else:
        print("Start training...")
        model.fit({'input': x_train}, {'targets': y_train}, n_epoch=50, show_metric=True, batch_size=32,
                  validation_set=({'input': x_test}, {'targets': y_test}))
        print("Finished...")
        model.save('model_6.tfl')

    return model


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
    conv_input = input_data(shape=[None, 100, 100, 3], name='input')

    conv1 = conv_2d(conv_input, 32, 3, activation='relu')
    pool1 = max_pool_2d(conv1, 2)

    conv2 = conv_2d(pool1, 32, 3, activation='relu')
    pool2 = max_pool_2d(conv2, 2)

    conv3 = conv_2d(pool2, 48, 3, activation='relu')
    pool3 = max_pool_2d(conv3, 2)

    conv4 = conv_2d(pool3, 48, 3, activation='relu')
    pool4 = max_pool_2d(conv4, 2)

    conv5 = conv_2d(pool4, 48, 3, activation='relu')
    pool5 = max_pool_2d(conv5, 2)

    conv6 = conv_2d(pool5, 32, 3, activation='relu')
    pool6 = max_pool_2d(conv6, 2)
    flat1 = flatten(pool6)

    fully_layer = fully_connected(flat1, 128, activation='relu')
    drop = dropout(fully_layer, 0.5)

    full_layer1 = fully_connected(drop, 64, activation='relu')
    drop1 = dropout(full_layer1, 0.5)

    cnn_layers = fully_connected(drop1, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(),
                            name='targets', to_one_hot=True, n_classes=6)

    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3, best_checkpoint_path='Best')
    print("Start training...")

    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=50, show_metric=True,
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

    final_layer = dropout(fully_layer3, 0.5)
    cnn_layers = fully_connected(final_layer, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(),
                            name='targets', to_one_hot=True, n_classes=6)
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    print("Start training...")
    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=50, show_metric=True,
              validation_set=({'input': x_test}, {'targets': y_test}))
    print("Finished...")
    return model


def model_4(x_train, x_test, y_train, y_test):
    conv_input = input_data(shape=[None, 100, 100, 3], name='input')

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

    final_layer = dropout(fully_layer3, 0.5)

    cnn_layers = fully_connected(final_layer, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(),
                            name='targets', to_one_hot=True, n_classes=6)
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    print("Start training...")
    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=100, show_metric=True,
              validation_set=({'input': x_test}, {'targets': y_test}))
    print("Finished...")
    return model


def model_5(x_train, x_test, y_train, y_test):
    conv_input = input_data(shape=[None, 227, 227, 3], name='input')

    conv1 = conv_2d(conv_input, 96, 11, activation='relu', strides=4)
    pool1 = max_pool_2d(conv1, 3, strides=2)
    norm1 = local_response_normalization(pool1)

    conv2 = conv_2d(norm1, 256, 5, activation='relu', strides=1)
    pool2 = max_pool_2d(conv2, 3, strides=2)
    norm2 = local_response_normalization(pool2)

    conv3 = conv_2d(norm2, 384, 3, activation='relu', strides=1)

    conv4 = conv_2d(conv3, 384, 3, activation='relu', strides=1)

    conv5 = conv_2d(conv4, 256, 3, activation='relu', strides=1)
    pool3 = max_pool_2d(conv5, 3, strides=2)
    flat1 = flatten(pool3)

    fully_layer1 = fully_connected(flat1, 4096, activation='relu')
    drop1 = dropout(fully_layer1, 0.5)

    fully_layer2 = fully_connected(drop1, 4096, activation='relu')
    drop2 = dropout(fully_layer2, 0.5)

    fully_layer3 = fully_connected(drop2, 1000, activation='relu')
    drop3 = dropout(fully_layer3, 0.5)

    cnn_layers = fully_connected(drop3, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(),
                            name='targets', to_one_hot=True, n_classes=6)
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    print("Start training...")
    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=100, show_metric=True, batch_size=32,
              validation_set=({'input': x_test}, {'targets': y_test}))
    print("Finished...")
    return model


def vgg_16(x_train, x_test, y_train, y_test):
    conv_input = input_data(shape=[None, 90, 90, 3], name='input')

    conv1 = conv_2d(conv_input, 32, 3, activation='relu')
    conv2 = conv_2d(conv1, 32, 3, activation='relu')

    pool1 = max_pool_2d(conv2, 2, strides=2)

    conv3 = conv_2d(pool1, 64, 3, activation='relu')
    conv4 = conv_2d(conv3, 64, 3, activation='relu')

    pool2 = max_pool_2d(conv4, 2, strides=2)

    conv5 = conv_2d(pool2, 128, 3, activation='relu')
    conv6 = conv_2d(conv5, 128, 3, activation='relu')

    pool3 = max_pool_2d(conv6, 2, strides=2)

    conv7 = conv_2d(pool3, 256, 3, activation='relu')
    conv8 = conv_2d(conv7, 256, 3, activation='relu')
    conv9 = conv_2d(conv8, 256, 3, activation='relu')

    pool4 = max_pool_2d(conv9, 2, strides=2)

    fully_layer1 = fully_connected(pool4, 4096, activation='relu')
    fully_layer2 = fully_connected(fully_layer1, 4096, activation='relu')
    fully_layer3 = fully_connected(fully_layer2, 1000, activation='relu')

    final_layer = dropout(fully_layer3, 0.5)

    cnn_layers = fully_connected(final_layer, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                            metric=Accuracy(), name='targets', to_one_hot=True, n_classes=6)
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    print("Start training...")

    if os.path.exists('model.meta'):
        model.load('./model')

    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=100, show_metric=True,
              validation_set=({'input': x_test}, {'targets': y_test}))
    model.save('model')

    print("Finished...")

    return model