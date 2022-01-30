import os
import datetime

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from utils import preprocess_input, plot_acc_and_loss_all
from keras.callbacks import ModelCheckpoint, CSVLogger
# from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping
import pickle

# --------------------------------------------------Global parameters--------------------------------------------------
plot = True  # If plot is true, the performance of the model will be shown (accuracy, loss, etc.)
backbone = 'EfficientNetB2'
num_of_experiment = '4'

# Paths to database
data_dir = ['../../../M4/MIT_small_train_1',
            '../../../M4/MIT_small_train_2',
            '../../../M4/MIT_small_train_3',
            '../../../M4/MIT_small_train_4']
train_data_dir = [s + '/train' for s in data_dir]
val_data_dir = [s + '/validation' for s in data_dir]
test_data_dir = [s + '/test' for s in data_dir]

# Image params
img_width = 224
img_height = 224

# NN params
batch_size = 16
number_of_epoch = 50
LR = 0.001
optimizer = tf.keras.optimizers.Adagrad(learning_rate=LR)

train_samples = 400
validation_samples = 807
test_samples = 807

# Experiment 2
freeze_layers = False  # If this variable is activated, we will freeze the layers of the base model to train parameters
# Experiment 3
train_again = False  # If this variable is activated, we will train again after the freezing the whole model.
# Experiment 4
new_layers = False  # Activate this variable to append new layers in between of the base model and the prediction layer
# ---------------------------------------------------------------------------------------------------------------------
i = 1
for (dir_train, dir_val, dir_test) in zip(train_data_dir, val_data_dir, test_data_dir):
    # Create the specific folders for this week, only if they don't exist
    print(f'iteration {i} of 4...')
    if not os.path.exists("models"):
        os.mkdir("models")

    date_start = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path_model = "models/" + backbone + '_exp_' + num_of_experiment + '_' + date_start
    if not os.path.exists(path_model):
        os.mkdir(path_model)
        os.mkdir(path_model + "/results")
        os.mkdir(path_model + "/saved_model")

    # Store description of experiment setup in a txt file
    with open(path_model + '/setup_description.txt', 'w') as f:
        f.write('Experiment set-up for: ' + path_model)
        f.write('\nExperiment number: ' + num_of_experiment)
        f.write('\nBackbone: ' + backbone)
        f.write('\nFreze Layers: ' + str(freeze_layers))
        f.write('\nBatch Norm + Relu: ' + str(new_layers))
        f.write('\nOptimizer: ' + str(optimizer))
        f.write('\nLearning Rate: ' + str(LR))
        f.write('\nTrain samples: ' + str(train_samples))
        f.write('\nValidation samples: ' + str(validation_samples))
        f.write('\nTest samples: ' + str(test_samples))
        f.write('\nBatch Size: ' + str(batch_size))

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
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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

    train_generator = datagen.flow_from_directory(dir_train,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

    test_generator = datagen.flow_from_directory(dir_val,
                                                 target_size=(img_width, img_height),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

    validation_generator = datagen.flow_from_directory(dir_test,
                                                       target_size=(img_width, img_height),
                                                       batch_size=batch_size,
                                                       class_mode='categorical')

    history = model.fit(train_generator,
                        steps_per_epoch=int(train_samples // batch_size),
                        epochs=number_of_epoch,
                        shuffle=True,
                        validation_data=validation_generator,
                        validation_steps=int(validation_samples // batch_size),
                        callbacks=[
                            # '/path_model + "/saved_model/"+backbone_epoch{epoch:02d}_acc{val_accuracy:.2f}'+backbone+'.h5'
                            ModelCheckpoint(path_model + "/saved_model/" + backbone + '.h5',
                                            monitor='val_accuracy',
                                            save_best_only=True,
                                            save_weights_only=True),
                            CSVLogger(
                                path_model + '/results/log_classification_' + backbone + '_exp_' + num_of_experiment + '.csv',
                                append=True, separator=';'),
                            # TensorBoard(path_model + '/tb_logs_' + backbone + '_exp_' + num_of_experiment, update_freq=1),
                            EarlyStopping(monitor='val_accuracy', patience=8, min_delta=0.001, mode='max')])

    path_history = 'models/' + 'history_' + str(i) + '.pickle'
    with open(path_history, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    result = model.evaluate(test_generator)
    print(result)

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

        path_history = 'models/' + 'history_' + str(i) + '_retrained.pickle'
        with open(path_history, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    i += 1


with open('models/history_1.pickle', "rb") as input_file:
    history1 = pickle.load(input_file)
with open('models/history_2.pickle', "rb") as input_file:
    history2 = pickle.load(input_file)
with open('models/history_3.pickle', "rb") as input_file:
    history3 = pickle.load(input_file)
with open('models/history_4.pickle', "rb") as input_file:
    history4 = pickle.load(input_file)
if train_again:
    with open('models/history_1_retrained.pickle', "rb") as input_file:
        history1_r = pickle.load(input_file)
    with open('models/history_2_retrained.pickle', "rb") as input_file:
        history2_r = pickle.load(input_file)
    with open('models/history_3_retrained.pickle', "rb") as input_file:
        history3_r = pickle.load(input_file)
    with open('models/history_4_retrained.pickle', "rb") as input_file:
        history4_r = pickle.load(input_file)

path_imgs = path_model + "/results"
plot_acc_and_loss_all(history1, history2, history3, history4, path_imgs)
if train_again:
    plot_acc_and_loss_all(history1_r, history2_r, history3_r, history4_r, path_imgs, retrained=True)

os.remove('models/history_1.pickle')
os.remove('models/history_2.pickle')
os.remove('models/history_3.pickle')
os.remove('models/history_4.pickle')
if train_again:
    os.remove('models/history_1_retrained.pickle')
    os.remove('models/history_2_retrained.pickle')
    os.remove('models/history_3_retrained.pickle')
    os.remove('models/history_4_retrained.pickle')
