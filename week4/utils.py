from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


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


def plot_acc_and_loss(history, path_model):
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
