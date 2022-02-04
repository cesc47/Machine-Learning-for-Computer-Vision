import os
import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from utils import preprocess_input, plot_acc_and_loss
from keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping
from ourCNN import customCNN

# --------------------------------------------------Global parameters--------------------------------------------------
plot = True  # If plot is true, the performance of the model will be shown (accuracy, loss, etc.)
backbone = 'CustomCNN'
num_of_experiment = '1'

# Paths to database
data_dir = '../../../M4/MIT_small_train_1'

train_data_dir = data_dir + '/train'
val_data_dir = data_dir + '/test'
test_data_dir = data_dir + '/validation'

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
# ---------------------------------------------------------------------------------------------------------------------

# Create the specific folders for this week, only if they don't exist
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
    f.write('\nOptimizer: ' + str(optimizer))
    f.write('\nLearning Rate: ' + str(LR))
    f.write('\nTrain samples: ' + str(train_samples))
    f.write('\nValidation samples: ' + str(validation_samples))
    f.write('\nTest samples: ' + str(test_samples))
    f.write('\nBatch Size: ' + str(batch_size))

# charge our model
model = customCNN(img_height, img_width, plot_summary=True)

file = path_model + "/saved_model" + '/completeModel.png'
plot_model(model, to_file=file, show_shapes=True, show_layer_names=True)

# Compile the model with its loss and optimizer and show what layers are going to be trained
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


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


result = model.evaluate(test_generator)
print(result)

plot_acc_and_loss(history, path_model)

