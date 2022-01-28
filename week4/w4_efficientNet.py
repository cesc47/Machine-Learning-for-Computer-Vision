import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

# --------------------------------------------------Global parameters--------------------------------------------------
plot = True  # If plot is true, the performance of the model will be shown (accuracy, loss, etc.)
model_used = 'EfficientNetB2'
num_of_experiment = '1'

# Paths to database
data_dir = '../../../M4/MIT_small_train_1'
train_data_dir = data_dir + '/train'
val_data_dir = data_dir + '/test'
test_data_dir = val_data_dir

# Image params
img_width = 224
img_height = 224

# NN params
batch_size = 32
number_of_epoch = 50
validation_samples = 807
# Experiment 2
freeze_layers = True  # If this variable is activated, we will freeze the layers of the base model to train parameters
# Experiment 3
train_again = True  # If this variable is activated, we will train again after the freezing the whole model.
# ---------------------------------------------------------------------------------------------------------------------

# Create the specific folders for this week, only if they don't exist
if not os.path.exists("models"):
    os.mkdir("models")

path_model = "models/" + model_used + '_' + num_of_experiment
if not os.path.exists(path_model):
    os.mkdir(path_model)
    os.mkdir(path_model + "/results")
    os.mkdir(path_model + "/saved_model")


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        # Returns the default image data format convention: A string, either 'channels_first' or 'channels_last'.
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]  # ::-1 => reverse of list
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:  # channels_last
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x


# create the base pre-trained model
base_model = tf.keras.applications.EfficientNetB2()

file = path_model + "/saved_model" + '/completeModel.png'
plot_model(base_model, to_file=file, show_shapes=True, show_layer_names=True)

# Take last layer and put a softmax to adapt the model to our problem (8 classes)
x = base_model.layers[-2].output
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
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    file = path_model + '/results' + '/accuracy.jpg'
    plt.savefig(file)
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    file = path_model + '/results' + '/loss.jpg'
    plt.savefig(file)
    plt.close()


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
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy RETRAINED')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        file = path_model + '/results' + '/accuracy_retrained.jpg'
        plt.savefig(file)
        plt.close()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss RETRAINED')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        file = path_model + '/results' + '/loss_retrained.jpg'
        plt.savefig(file)
        plt.close()

model_f_path = path_model + "/saved_model" + '/' + model_used + '_' + num_of_experiment + '.h5'
model.save_weights(model_f_path)  # always save your weights after training or during training

