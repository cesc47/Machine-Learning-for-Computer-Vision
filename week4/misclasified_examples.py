import numpy as np
import tensorflow as tf
import glob
import cv2
from utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_width = 224
img_height = 224

data_dir = '../../../M4/MIT_small_train_1'
test_data_dir = data_dir + '/test/'

model = tf.keras.models.load_model(r'C:\Users\Cesc47\PycharmProjects\MCV\M4\Machine-Learning-for-Computer-Vision\week4\models\EfficientNetB2_exp_5_2022-01-30-18-18-57\saved_model\EfficientNetB25.h5')

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             preprocessing_function=preprocess_input,
                             rotation_range=0.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=None)

test_generator = datagen.flow_from_directory(test_data_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=1,
                                             shuffle=False,
                                             class_mode='categorical')

gtruths = test_generator.classes
preds = np.argmax(model.predict(test_generator), axis=-1)
indices = test_generator.class_indices
filenames = test_generator.filenames
filepaths = test_generator.filepaths
it = 0
fallos = 0
for pred in preds:
    gtruth = gtruths[it]
    if pred != gtruth:
        fallos += 1
        print(f'Misclasified img!\tfile:{test_generator.filenames[it]}\tclass {gtruth} but classified into class:{pred}.\terrors:{fallos}')
    it += 1


"""
it = 1
fallos = 0
for direction in test_data_dir:
    path = direction + '/*.jpg'
    files = [file for file in glob.glob(path)]
    for file in files:
        img = cv2.imread(file)
        img = img.astype('float64')
        img = preprocess_input(img)
        img = cv2.resize(img, (260, 260))  # resize image to match model's expected sizing
        img = img.reshape(1, 260, 260, 3)
        pred = model.predict(img)
        if int(np.argmax(pred)) != it:
            fallos += 1
            print(f'Misclasified img!\tfile:{file}\tclass {it} but classified into class:{np.argmax(pred)}.\terrors:{fallos}')
    it += 1
"""
