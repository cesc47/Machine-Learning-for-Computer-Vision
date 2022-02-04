import keras
import numpy as np
from keras.models import Sequential
from keras import layers

# Very basic model
def customCNN(img_width, img_height, plot_summary=False):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))
    if plot_summary:
        model.summary()

    return model


def customCNNv2(img_width, img_height, plot_summary=False):
    model = Sequential()


# IDEAS:
# Empezar con n_h, n_w grande para acabar con n_c grande? Demostrar que mas capas, mas accuracy.
# Modelo con skip connections => modelo secuencial no vale supongo
# conv 1x1 para reducir n_c
# Prunar modelo al final. mecanismo iterativo q busque la mejor solucion
