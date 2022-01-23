from __future__ import print_function

import pickle

from keras.layers import Dense, Reshape
from keras.models import Sequential, Model
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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
descriptors = np.empty((len(train_images_filenames), num_patches, model.layers[-1].output_shape[1]))
for i, filename in enumerate(train_images_filenames):
    img = Image.open(filename)
    patches = image.extract_patches_2d(np.array(img), (PATCH_SIZE, PATCH_SIZE), max_patches=num_patches)
    descriptors[i, :, :] = model.predict(patches / 255.)
features = np.vstack(descriptors)
print('Features extracted.')

codebook = MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                           reassignment_ratio=10 ** -4, random_state=42)
codebook.fit(features)
print('codebook generated.')

visual_words = np.empty((len(descriptors), k), dtype=np.float32)
for i, des in enumerate(descriptors):
    words = codebook.predict(des)
    visual_words[i, :] = np.bincount(words, minlength=k)

visual_words_train = StandardScaler().fit_transform(visual_words)
print('visual words trained.')

# gridsearch SVM
param_grid = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01], 'C': [0.1, 1, 10]},
              {'kernel': ['linear'], 'C': [0.1, 1, 10]},
              {'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [1, 0.1, 0.01], 'C': [0.1, 1, 10]}]

grid = GridSearchCV(svm.SVC(), param_grid, n_jobs=3, cv=8)
grid.fit(visual_words_train, train_labels)

best_kernel = grid.best_params_['kernel']
print(f'Best SVM kernel: {best_kernel}')

# test
descriptors_test = np.empty((len(train_images_filenames), num_patches, model.layers[-1].output_shape[1]))
for i, filename in enumerate(test_images_filenames):
    img = Image.open(filename)
    patches = image.extract_patches_2d(np.array(img), (PATCH_SIZE, PATCH_SIZE), max_patches=num_patches)
    descriptors_test[i, :, :] = model.predict(patches / 255.)
print('prediction of the test descriptors done.')

visual_words = np.empty((len(descriptors_test), k), dtype=np.float32)
for i, des in enumerate(descriptors_test):
    words = codebook.predict(des)
    visual_words[i, :] = np.bincount(words, minlength=k)
print('prediction of the visual words done.')

visual_words_test = StandardScaler().fit_transform(visual_words)

classifier = svm.SVC(kernel=best_kernel)
classifier.fit(visual_words_test, train_labels)

accuracy = classifier.score(visual_words_test, test_labels)

print(f'Test accuracy: {accuracy}')
