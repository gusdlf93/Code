import os
#if using resnet18
os.chdir(("/").join(__file__.split('/')[:-1]))

import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from Model import E_Model

def parsing_data(path):
    f = open(path, 'r')
    data = []
    label = []
    lines = f.readlines()
    for line in lines:
        x = line.split(' ')
        x = [float(x[i]) for i in range(len(x) - 1)]
        data.append(x[:-1])
        label.append(int(x[-1]))
    return data, label

def normalize(x_train, x_test):

    for i in range(len(x_train)):
        min = np.abs(np.min(x_train[i]))
        x_train[i] = x_train[i] + min
        max = np.max(x_train[i])
        x_train[i] = (x_train[i] / max) * 255

    for i in range(len(x_test)):
        min = np.abs(np.min(x_test[i]))
        x_test[i] = x_test[i] + min
        max = np.max(x_test[i])
        x_test[i] = (x_test[i] / max) * 255

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = x_train.reshape((len(x_train), 160, 200))#2000, 16은 너무 꺠져버림
    x_test = x_test.reshape((len(x_test), 160, 200))

    train = []
    for i in range(len(x_train)):
        train.append(cv2.resize(x_train[i], (224, 224)).astype('int32'))

    test = []
    for i in range(len(x_test)):
        test.append(cv2.resize(x_test[i], (224, 224)).astype('int32'))

    return train, test

train_datagen = ImageDataGenerator(
    rescale=1./255
)

Models = {0 : 'basic_model_5layer', 1 : 'basic_model_10layer', 2 : 'basic_model_v2', 3: 'basic_model_v3', 4: 'LeNet', 5: 'Resnet50', 6 : 'Resnet18'}
Model_num  = 3
if Model_num < 5: #Resnet을 제외한 모델은 (16, 2000, 1)형태의 이미지를 Input으로
    root = r'D:\e-nose\rectangle\original'
else :# Resnet은 (224, 224, 3) 이미지로  resized되서 학습됩니다.
    root = r'D:\e-nose\square\original'


root_folder = os.listdir(root)

for _, i in enumerate(root_folder):# i == iter
    first_path = os.path.join(root, i)

    first_folder = os.listdir(first_path)

    for __, j in enumerate(first_folder): # j == set
        print(_*len(root_folder) + __)

        second_path = os.path.join(first_path, j)

        second_folder = os.listdir(second_path)

        model = E_Model(Models[Model_num])
        color = model.color
        input_shape = model.input_shape

        model = model.get_model()
        model.summary()

        sgd = keras.optimizers.SGD(lr=0.001, decay=0.005, momentum=0.9)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        train_generator = train_datagen.flow_from_directory(
            os.path.join(root, i, j, j + '_tr_data'),#원본 데이터로 학습
            shuffle=True,
            #target_size=(input_shape[0], input_shape[1]),
            target_size=(input_shape[0], input_shape[1]),
            batch_size=16,
            color_mode=color,
            class_mode='categorical')

        model.fit_generator(
            train_generator,
            steps_per_epoch= 140 // 16,
            validation_data=None,
            validation_steps=None,
            epochs=50, verbose=1)

        folder = os.path.join(r"./weight", Models[Model_num])
        if not os.path.exists(folder):
            os.makedirs(folder)
        model.save_weights(os.path.join(folder, r"%d_%d_e-nose_1D.h5"% (_+1, __+1)))

        keras.backend.clear_session()
