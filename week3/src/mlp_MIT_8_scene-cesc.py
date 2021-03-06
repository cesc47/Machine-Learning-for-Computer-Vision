import cv2
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras.layers import Dense, Reshape
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

from utils import *

f = open("env.txt", "r")
ENV = f.read().split('"')[1]

# user defined variables
IMG_SIZE = 32
BATCH_SIZE = 8
if ENV == "local":
    DATASET_DIR = '../../../MIT_split'
    MODEL_FNAME = './models/MLP1'

else:
    DATASET_DIR = '/home/mcv/datasets/MIT_split'
    MODEL_FNAME = './models/my_first_mlp.h5'

if not os.path.exists(MODEL_FNAME):
    os.makedirs(MODEL_FNAME)

if not os.path.exists(DATASET_DIR):
    print(Color.RED, 'ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
    quit()

# Build the Multi Layer Perceptron model
model = Sequential()
model.add(Reshape((IMG_SIZE * IMG_SIZE * 3,), input_shape=(IMG_SIZE, IMG_SIZE, 3), name='first'))
model.add(BatchNormalization())
model.add(Dense(units=2048, activation='relu', name='second'))
model.add(Dense(units=4092, kernel_regularizer='l2', name='third'))
model.add(Dense(units=8192, activation='tanh', name='fourth'))
model.add(Dense(units=4092, kernel_regularizer='l2', name='fifth'))
model.add(Dense(units=2048, activation='relu', name='sixth'))
model.add(Dense(units=1024, name='seventh'))
model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adagrad',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)


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
    target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
    batch_size=BATCH_SIZE,
    classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    DATASET_DIR + '/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    class_mode='categorical')

history = model.fit(
    train_generator,
    steps_per_epoch=1881 // BATCH_SIZE,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=807 // BATCH_SIZE)

print('End of training!\n')
print('Saving the model into ' + MODEL_FNAME + ' \n')
model.save(MODEL_FNAME)  # always save your weights after training or during training

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.jpg')
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.jpg')

# to get the output of a given layer
# crop the model up to a certain layer
model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)

# get the features from images
directory = DATASET_DIR + '/test/coast'
x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0])))
x = np.expand_dims(cv2.resize(x, (IMG_SIZE, IMG_SIZE)), axis=0)
print('prediction for image ' + os.path.join(directory, os.listdir(directory)[0]))
features = model_layer.predict(x / 255.0)
print(features)
