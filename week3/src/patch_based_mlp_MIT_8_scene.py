from __future__ import print_function

import matplotlib.pyplot as plt
from keras.layers import Dense, Reshape
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

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

model_name = 'exp9_basic_patch'
model_path = "models/" + model_name + "/"

model_f_path = model_path + model_name + '.h5'

if not os.path.exists(model_path):
    os.mkdir(model_path)

if not os.path.exists(DATASET_DIR):
    colorprint(Color.RED, 'ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
    quit()

if not os.path.exists(PATCHES_DIR):
    generate_image_patches_db(DATASET_DIR, PATCHES_DIR, patch_size=PATCH_SIZE)

colorprint(Color.BLUE, 'Building MLP model...\n')
model = build_mlp(input_size=PATCH_SIZE)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
print(model.summary())

# if os.path.exists(model_f_path):
#     model.load_weights(model_f_path)
#     colorprint(Color.YELLOW, 'WARNING: model file ' + model_f_path + ' exists! Loading weights\n')

colorprint(Color.BLUE, 'Start training...\n')
# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
dir_patches = PATCHES_DIR + "_" + str(PATCH_SIZE)
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    dir_patches + '/train',  # this is the target directory
    target_size=(PATCH_SIZE, PATCH_SIZE),  # all images will be resized to PATCH_SIZExPATCH_SIZE
    batch_size=BATCH_SIZE,
    classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    dir_patches + '/test',
    target_size=(PATCH_SIZE, PATCH_SIZE),
    batch_size=BATCH_SIZE,
    classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    class_mode='categorical')

history = model.fit(
    train_generator,
    steps_per_epoch=18810 // BATCH_SIZE,
    epochs=100,
    shuffle=True,
    validation_data=validation_generator,
    validation_steps=8070 // BATCH_SIZE)

output_path = model_path + model_name + '_history'
np.save(output_path, history.history)
plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)
print('Finished Training\n')
print('Saving the model into ' + model_f_path + ' \n')
model.save_weights(model_f_path)  # always save your weights after training or during training

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(model_path + 'accuracy.jpg')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(model_path + 'loss.jpg')

colorprint(Color.BLUE, 'Building MLP model for testing...\n')
model = build_mlp(input_size=PATCH_SIZE, phase='TEST')
print(model.summary())

colorprint(Color.BLUE, 'Loading weights from ' + model_f_path + ' ...\n')
model.load_weights(model_f_path)

colorprint(Color.BLUE, 'Start evaluation ...\n')

directory = DATASET_DIR+'/test'
classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
correct = 0.
total   = 807
count   = 0

for class_dir in os.listdir(directory):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory,class_dir)):
      im = Image.open(os.path.join(directory,class_dir,imname))
      patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE),
                                         max_patches=int(np.asarray(im).shape[0] / PATCH_SIZE) ** 2)
      out = model.predict(patches/255.)

      # Compute mean of all the classes and then apply softmax:
      #predicted_cls = np.argmax( softmax(np.mean(out,axis=0)) )

      # Compute softmax for each patch and then assign the most voted patch:
      predictions = np.argmax(softmax(out), axis=-1)
      predicted_cls = np.argmax(np.bincount(predictions))

      if predicted_cls == cls:
        correct+=1
      count += 1
      print('Evaluated images: '+str(count)+' / '+str(total), end='\r')

colorprint(Color.BLUE, 'Done!\n')
colorprint(Color.GREEN, 'Test Acc. = '+str(correct/total)+'\n')
