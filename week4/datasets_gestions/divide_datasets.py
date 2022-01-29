from pathlib import Path
import numpy as np
import os
from PIL import Image
import cv2
import glob

# generate test dataset from the MIT splits
dataset_nums = ['1', '2',  '3', '4']
data_dir = '../../../MIT_small_train_'
data_dir = [data_dir + num for num in dataset_nums]
test_data_dir = [direc + '/test/' for direc in data_dir]
validation_data_dir = [direc + '/validation/' for direc in data_dir]

classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

directions_test = []
for data_dir in test_data_dir:
    directions_test = np.append(directions_test, [data_dir + s for s in classes])

directions_val = []
for data_dir in validation_data_dir:
    directions_val = np.append(directions_val, [data_dir + s for s in classes])

for directory in validation_data_dir:
    if not os.path.exists(directory):
        os.mkdir(directory)

for directory in directions_val:
    if not os.path.exists(directory):
        os.mkdir(directory)

for direction in directions_test:
    path = direction + '/*.jpg'
    files = [file for file in glob.glob(path)]
    it = 0
    print(files)
    for file in files:
        if it < len(files) / 2:
            val_dir = file.replace('test', 'validation')
            img = Image.open(file)
            img = img.save(val_dir)
            os.remove(file)
            it += 1