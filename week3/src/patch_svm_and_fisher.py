from __future__ import print_function

import pickle

from keras.layers import Dense, Reshape
from keras.models import Sequential, Model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from utils import *

# user defined variables
PATCH_SIZE = 64
BATCH_SIZE = 16


def build_mlp(input_size=PATCH_SIZE, phase='TRAIN'):
    model = Sequential()
    model.add(Reshape((input_size * input_size * 3,), input_shape=(input_size, input_size, 3)))
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    if phase == 'TEST':
        model.add(
            Dense(units=8, activation='linear'))  # In test phase we softmax the average output over the image patches
    else:
        model.add(Dense(units=8, activation='softmax'))
    return model


f = open("env.txt", "r")
ENV = f.read().split('"')[1]

DATASET_DIR = '../../../MIT_split'
DATA_DIR = 'data/'

PATCHES_DIR = DATA_DIR + 'MIT_split_patches'

model_name = 'exp8_basic_patch'
model_path = "models/" + model_name + "/"

model_f_path = model_path + model_name + '.h5'

if not os.path.exists(model_path):
    os.mkdir(model_path)

if not os.path.exists(DATASET_DIR):
    colorprint(Color.RED, 'ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
    quit()

model = build_mlp(input_size=PATCH_SIZE)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

colorprint(Color.BLUE, 'Building MLP model for testing...\n')

model = build_mlp(input_size=PATCH_SIZE, phase='TEST')
print(model.summary())

model.load_weights(model_f_path)

model = Model(inputs=model.input, outputs=model.layers[-2].output)
model.summary()

k = 800
im_size = 256
train_images_filenames = pickle.load(open(DATASET_DIR + '/train_images_filenames.dat', 'rb'))
test_images_filenames = pickle.load(open(DATASET_DIR + '/test_images_filenames.dat', 'rb'))
train_images_filenames = [DATASET_DIR + '/' + n[26:] for n in train_images_filenames]
test_images_filenames = [DATASET_DIR + '/' + n[26:] for n in test_images_filenames]

train_labels = pickle.load(open(DATASET_DIR + '/train_labels.dat', 'rb'))
test_labels = pickle.load(open(DATASET_DIR + '/test_labels.dat', 'rb'))

print(f'{len(train_images_filenames)} training images')
print(f'{len(test_images_filenames)} test images')

Train_descriptors = []
Train_label_per_descriptor = []
num_patches = int((im_size / PATCH_SIZE) ** 2)

# train
descriptors_train = np.empty((len(train_images_filenames), num_patches, model.layers[-1].output_shape[1]))
for i, filename in enumerate(train_images_filenames):
    img = Image.open(filename)
    patches = image.extract_patches_2d(np.array(img), (PATCH_SIZE, PATCH_SIZE), max_patches=num_patches)
    descriptors_train[i, :, :] = model.predict(patches / 255.)
features = np.vstack(descriptors_train)
print('Features train extracted.')

# test
descriptors_test = np.empty((len(train_images_filenames), num_patches, model.layers[-1].output_shape[1]))
for i, filename in enumerate(test_images_filenames):
    img = Image.open(filename)
    patches = image.extract_patches_2d(np.array(img), (PATCH_SIZE, PATCH_SIZE), max_patches=num_patches)
    descriptors_test[i, :, :] = model.predict(patches / 255.)
features_test = np.vstack(descriptors_test)
print('Features test extracted.')

FV_train = train_fv(features, descriptors_train)
print('Fisher vectors train generated.')

FV_test = train_fv(features, descriptors_test)
print('Fisher vectors test generated.')

# gridsearch SVM
param_grid = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01], 'C': [0.1, 1, 10]},
              {'kernel': ['linear'], 'C': [0.1, 1, 10]},
              {'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [1, 0.1, 0.01], 'C': [0.1, 1, 10]},
              {'kernel': [intersection_kernel]}]

grid = GridSearchCV(svm.SVC(), param_grid, n_jobs=3, cv=8)
grid.fit(FV_train, train_labels)

best_kernel = grid.best_params_['kernel']
print(f'Best SVM kernel: {best_kernel}')

grid_predictions = grid.predict(FV_test)
print("classification_report\n", classification_report(test_labels, grid_predictions))

