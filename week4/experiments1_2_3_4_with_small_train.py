import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from utils import preprocess_input, plot_acc_and_loss
# --------------------------------------------------Global parameters--------------------------------------------------
plot = True  # If plot is true, the performance of the model will be shown (accuracy, loss, etc.)
model_used = 'EfficientNetB2'
num_of_experiment = '1'

# Paths to database
data_dir = ['../../../M4/MIT_small_train_1',
            '../../../M4/MIT_small_train_2',
            '../../../M4/MIT_small_train_3',
            '../../../M4/MIT_small_train_4']
train_data_dir = [s + '/train' for s in data_dir]
val_data_dir = [s + '/test' for s in data_dir]
test_data_dir = val_data_dir

# Image params
img_width = 224
img_height = 224

# NN params
batch_size = 32
number_of_epoch = 2
validation_samples = 807
# Experiment 2
freeze_layers = True  # If this variable is activated, we will freeze the layers of the base model to train parameters
# Experiment 3
train_again = True  # If this variable is activated, we will train again after the freezing the whole model.
# Experiment 4
new_layers = True  # Activate this variable to append new layers in between of the base model and the prediction layer


# TO DO...:
# CALLBACKS
# HYPERPARAMETER SEARCH
# DATA AUGMENTATION
# TRY THE 4 DATASETS OF MIT_SMALL_TRAIN
# mas propuestas...

# ---------------------------------------------------------------------------------------------------------------------

# Create the specific folders for this week, only if they don't exist
if not os.path.exists("models"):
    os.mkdir("models")

path_model = "models/" + model_used + '_' + num_of_experiment
if not os.path.exists(path_model):
    os.mkdir(path_model)
    os.mkdir(path_model + "/results")
    os.mkdir(path_model + "/saved_model")

# create the base pre-trained model
base_model = tf.keras.applications.EfficientNetB2()

file = path_model + "/saved_model" + '/completeModel.png'
plot_model(base_model, to_file=file, show_shapes=True, show_layer_names=True)

# Take last layer and put a softmax to adapt the model to our problem (8 classes)
x = base_model.layers[-2].output
if new_layers:
    print('Appending new layers to the model after the base_model...')
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.5)(x)
    x = Dense(1024, activation='relu', name='extraProcessing1')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.2)(x)
    x = Dense(256, activation='relu', name='extraProcessing2')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

# Generate model from the base_model and our new generated layer
model = Model(inputs=base_model.input, outputs=x)
file = path_model + "/saved_model" + '/OurModel.png'
plot_model(model, to_file=file, show_shapes=True, show_layer_names=True)

# Freeze the layers from the model and train only the last one
if freeze_layers:
    print('Freezing layers from the model and training only the last one...')
    for layer in base_model.layers:
        layer.trainable = False

# Compile the model with its loss and optimizer and show what layers are going to be trained
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
for layer in model.layers:
    print(layer.name, layer.trainable)

# Apply the preprocess when training and testing (preprocessing_function=preprocess_input):
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

train_generator = datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

history = model.fit(train_generator,
                    steps_per_epoch=(int(400 // batch_size) + 1),
                    epochs=number_of_epoch,
                    validation_data=validation_generator,
                    validation_steps=(int(validation_samples // batch_size) + 1), callbacks=[])

result = model.evaluate(test_generator)
print(result)

if plot:
    plot_acc_and_loss(history, path_model)

if freeze_layers and train_again:
    tf.keras.backend.clear_session()
    print('Training again the whole model...')
    K.set_value(model.optimizer.learning_rate, 0.0001)  # Decrease learning rate (default - opt:Adam => 0.001)
    for layer in model.layers:
        layer.trainable = True
        print(layer.name, layer.trainable)

    history = model.fit(train_generator,
                        steps_per_epoch=(int(400 // batch_size) + 1),
                        epochs=number_of_epoch,
                        validation_data=validation_generator,
                        validation_steps=(int(validation_samples // batch_size) + 1), callbacks=[])

    if plot:
        plot_acc_and_loss(history, path_model)

model_f_path = path_model + "/saved_model" + '/' + model_used + '_' + num_of_experiment + '.h5'
model.save_weights(model_f_path)  # always save your weights after training or during training
