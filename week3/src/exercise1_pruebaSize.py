from utils import *
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# user defined variables
IMG_SIZE = [8, 16, 32, 64, 128]
BATCH_SIZE = 16

for size in IMG_SIZE:
    DATASET_DIR = '../../../MIT_split'
    MODEL_FNAME = './models/MLP' + str(size)

    if not os.path.exists(MODEL_FNAME):
        os.makedirs(MODEL_FNAME)

    if not os.path.exists(DATASET_DIR):
        print(Color.RED, 'ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
        quit()

    # Build the Multi Layer Perceptron model
    model = Sequential()
    model.add(Reshape((size * size * 3,), input_shape=(size, size, 3), name='first'))
    model.add(Dense(units=2048, activation='relu', name='second'))
    # model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    print(model.summary())
    # plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

    if os.path.exists(MODEL_FNAME):
        print('WARNING: model file ' + MODEL_FNAME + ' exists and will be overwritten!\n')

    print('Start training...\n')

    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True)

    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR + '/train',  # this is the target directory
        target_size=(size, size),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR + '/test',
        target_size=(size, size),
        batch_size=BATCH_SIZE,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    history = model.fit(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=75,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE)

    print('End of training!\n')
    print('Saving the model into ' + MODEL_FNAME + ' \n')
    model.save(MODEL_FNAME)  # always save your weights after training or during training

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])

    # to get the output of a given layer
    # crop the model up to a certain layer

    model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)


plt.legend(['Image size: 8x8', 'Image size: 16x16', 'Image size: 32x32', 'Image size: 64x64', 'Image size: 128x128'], loc='upper left')
plt.title('model accuracy (validation set)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('accuracy.jpg')
plt.close()