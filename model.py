
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU

from keras.layers import Convolution2D, BatchNormalization, AveragePooling2D, Dropout, Flatten, Dense, Activation

def simple_CNN(input_shape, num_classes):
    model = Sequential()
    model.add(Convolution2D(16, 7, 7, padding='same', input_shape=input_shape))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(5, 5),strides=(2, 2),padding='same'))
    model.add(Dropout(.5))



    model.add(Convolution2D(32, 5, 5, padding='same', input_shape=input_shape))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2), padding='same'))
    model.add(Dropout(.5))



    model.add(Convolution2D(32, 3, 3, padding='same', input_shape=input_shape))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(.5))

    # 展平
    model.add(Flatten())
    model.add(Dense(1028))
    model.add(PReLU())
    model.add(Dropout(.5))
    model.add(Dense(1028))
    model.add(PReLU())
    model.add(Dropout(.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
if __name__ == '__main__':
    input_shape = (48, 48, 1)
    num_classes = 7
    model = simple_CNN(input_shape, num_classes)
    model.summary()
