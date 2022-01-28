import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import os

# --------------------------------------------------Global parameters--------------------------------------------------
plot = True                 # If plot is true, the performance of the model will be shown (accuracy, loss, etc.)
model_used = 'VGG16'
num_of_experiment = '1'

# Paths to database
data_dir = '../../../M4/MIT_split'
train_data_dir = data_dir + '/train'
val_data_dir = data_dir + '/test'
test_data_dir = val_data_dir

# Image params
img_width = 224
img_height = 224

# NN params
batch_size = 32
number_of_epoch = 2
validation_samples = 807
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
        x = x[::-1, :, :] # ::-1 => reverse of list
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else: # channels_last
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x


# create the base pre-trained model
base_model = VGG16(weights='imagenet')
file = path_model + "/saved_model" + '/completeModel.png'
plot_model(base_model, to_file=file, show_shapes=True, show_layer_names=True)

# Take last layer and put a softmax to adapt the model to our problem (8 classes)
x = base_model.layers[-2].output
x = Dense(8, activation='softmax', name='predictions')(x)

# Freeze the layers from the model and train only the last one
model = Model(inputs=base_model.input, outputs=x)
file = path_model + "/saved_model" + '/OurModel.png'
plot_model(model, to_file=file, show_shapes=True, show_layer_names=True)
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

model_f_path = path_model + "/saved_model" + '/' + model_used + '_' + num_of_experiment + '.h5'
model.save_weights(model_f_path)  # always save your weights after training or during training

# list all data in history
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
