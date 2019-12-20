import os

from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Activation , pooling, Lambda, Conv1D, Input, GlobalAveragePooling2D, Convolution2D, AveragePooling2D
from keras.backend import mean
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras_radam import RAdam
from resnet import ResnetBuilder

def basic_model_10layer(input_shape, target_class):
    model = Sequential()
    channel_size = 8

    # encoding phase 1
    model.add(Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    channel_size *= 2
    model.add(Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    channel_size *= 2
    model.add(Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    channel_size *= 2
    model.add(Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # encoding phase 2
    channel_size *= 2
    model.add(Conv2D(channel_size, kernel_size=(3, 1), strides=(2, 1), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(channel_size, kernel_size=(3, 1), strides=(2, 1), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(channel_size, kernel_size=(3, 1), strides=(2, 1), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(channel_size, kernel_size=(3, 1), strides=(2, 1), padding='same'))

    model.add(Flatten())
    model.add(Dense(125, activation='relu'))

    # output layer
    model.add(Dense(target_class, activation='softmax'))

    return model

def basic_model_5layer(input_shape, target_class):
    input = Input(shape=input_shape)
    # encoding phase 1
    channel_size = 8
    x = Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), input_shape=input_shape, batch_size=16, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    channel_size*=2
    x = Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    channel_size *= 2
    x = Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    channel_size *= 2
    x = Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Lambda(lambda x: mean(x, axis=1))(x)  # keras.backend.sum은 학습이 이뤄지지 않습니다.

    # output layer
    x = Flatten()(x)
    output = Dense(target_class, activation='softmax')(x)

    model = Model(input, output)
    return model

def basic_model_v2(input_shape, target_class):
    input = Input(shape=input_shape)
    channel_size = 4

    # encoding phase 1
    x = Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), input_shape=input_shape, batch_size=16, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    channel_size *= 2
    x = Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    channel_size *= 2
    x = Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    channel_size *= 2
    x = Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Lambda(lambda x: mean(x, axis=1))(x)  # keras.backend.sum은 학습이 이뤄지지 않습니다.

    # output layer
    x = Flatten()(x)
    output = Dense(target_class, activation='softmax')(x)

    model = Model(input, output)
    return model

def basic_model_v3(input_shape, target_class):
    input = Input(shape=input_shape)
    channel_size = 2

    # encoding phase 1
    x = Conv2D(2, kernel_size=(1, 3), strides=(1, 2), input_shape=input_shape, batch_size=16, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    channel_size *= 2
    x = Conv2D(4, kernel_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    channel_size *= 2
    x = Conv2D(8, kernel_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    channel_size *= 2
    x = Conv2D(16, kernel_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Lambda(lambda x: mean(x, axis=1))(x)  # keras.backend.sum은 학습이 이뤄지지 않습니다.

    # output layer
    x = Flatten()(x)
    output = Dense(target_class, activation='softmax')(x)

    model = Model(input, output)
    return model

def Resnet50(input_shape, target_class):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet', input_tensor=None,
                          pooling=None)

    for layer in base_model.layers:  # 흑백 이미지라서 weight는 초기화 용으로만 사용합니다.
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    preds = Dense(target_class, activation='softmax')(x)  # final layer with softmax activation

    model = Model(inputs=base_model.input, outputs=preds)
    return model

def Resnet50_v2(input_shape, target_class):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights=None, input_tensor=None,
                          pooling=None)

    for layer in base_model.layers:  # 흑백 이미지라서 weight는 초기화 용으로만 사용합니다.
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    preds = Dense(target_class, activation='softmax')(x)  # final layer with softmax activation

    model = Model(inputs=base_model.input, outputs=preds)
    return model

def NasnetMobile(input_shape, target_class):
    base_model = NASNetMobile(input_shape=input_shape, include_top=False, weights='imagenet', input_tensor=None,
                              pooling=None)

    for layer in base_model.layers:  # 흑백 이미지라서 weight는 초기화 용으로만 사용합니다.
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    preds = Dense(target_class, activation='softmax')(x)  # final layer with softmax activation

    model = Model(inputs=base_model.input, outputs=preds)
    return model

def vgg16(input_shape, target_class):
    base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet', input_tensor=None,
                       pooling=None)

    for layer in base_model.layers:  # 흑백 이미지라서 weight는 초기화 용으로만 사용합니다.
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    preds = Dense(target_class, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)
    model.summary()

    return model

def Resnet18(input_shape, target_class):
    model = ResnetBuilder.build_resnet_18(input_shape, target_class)
    return model

def Alexnet(input_shape, target_class):
    model = Sequential()

    model.add(
        Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape,
               padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=(5, 5), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(384, kernel_size=(3, 3), strides=1, activation='relu'))

    model.add(Conv2D(384, kernel_size=(3, 3), strides=1, activation='relu'))

    model.add(Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1024, activation='relu'))

    # output layer
    model.add(Dense(target_class, activation='softmax'))

    return model

def LeNet(input_shape, target_class):
    model = Sequential()
    model.add(Convolution2D(6, kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=(input_shape), padding="same"))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
    model.add(Convolution2D(16, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid', data_format='channels_first'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(Convolution2D(120, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(target_class, activation='softmax'))

    return model
class E_Model():

    def __init__(self, target_model):
        self.target_model = target_model
        self.target_calss = 8

        self.Models = {'basic_model_5layer': basic_model_5layer, 'basic_model_10layer': basic_model_10layer, 'basic_model_v2' : basic_model_v2, 'basic_model_v3' : basic_model_v3,
                       'Resnet50': Resnet50, 'Resnet50_v2':Resnet50_v2, 'Resnet18': Resnet18, 'Resnet18_v2': Resnet18,
                       "LeNet":LeNet, "LeNet_v2":LeNet}

        if self.target_model == 'Resnet50' or self.target_model == 'Resnet18' or self.target_model == 'NasnetMobile' or self.target_model == 'Alexnet' or self.target_model == 'vgg16':
            self.color = 'rgb'
            self.input_shape = (224, 224, 3)
        elif self.target_model == "LeNet_v2" or self.target_model == "Resnet50_v2" or self.target_model == 'Resnet18_v2':
            self.color = 'grayscale'
            self.input_shape = (48, 2000, 1)
        else :
            self.color = 'grayscale'
            self.input_shape = (16, 2000, 1)


    def get_model(self):
        return self.Models[self.target_model](self.input_shape, self.target_calss)
