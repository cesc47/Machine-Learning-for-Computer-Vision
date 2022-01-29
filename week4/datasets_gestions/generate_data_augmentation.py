# Executing this file it will implement data augmentation in the /train directory and it will store it in another
# directory called train_augmented

import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob

# Paths to database
data_dir = '../../../MIT_small_train_1'
train_data_dir = data_dir + '/train/'
train_data_dir_augmented = data_dir + '/t_augmented/'

classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
directions_train = [train_data_dir + s for s in classes]
directions_train_augmented = [train_data_dir_augmented + s for s in classes]

if not os.path.exists(train_data_dir_augmented):
    os.mkdir(train_data_dir_augmented)

for directory in directions_train_augmented:
    if not os.path.exists(directory):
        os.mkdir(directory)

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

for direction in directions_train:
    print(f'actual direction: {direction}')
    data_augmented_dir = direction.replace('train/', 't_augmented/')
    data_augmented_dir = data_augmented_dir + '/'
    path = direction + '/*.jpg'
    files = [file for file in glob.glob(path)]

    for path_file in files:
        img = load_img(path_file)  # this is a PIL image
        prefix = path_file.split('\\')[1]
        prefix = prefix.replace('.jpg', '')
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=data_augmented_dir, save_prefix=prefix, save_format='jpg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely


