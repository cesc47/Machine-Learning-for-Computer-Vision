import cv2
from pathlib import Path
import numpy as np

# Comprobar que para la dataset den los valores que nos proporcionaron

data_dir = '../../../MIT_split'
train_data_dir = data_dir + '/train/'
test_data_dir = data_dir + '/test/'

classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
directions = [train_data_dir + s for s in classes]
directions_test = [test_data_dir + s for s in classes]

mean_b = 0
mean_g = 0
mean_r = 0
std_b = 0
std_g = 0
std_r = 0

it = 0
for dir in directions:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1


for dir in directions_test:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1

mean_b /= it
mean_g /= it
mean_r /= it
std_b /= it
std_g /= it
std_r /= it

print(f"Mit_split:")
print(f"MEAN:\tb:{mean_b}, g:{mean_g}, r:{mean_r}")
print(f"STD:\tb:{std_b}, g:{std_g}, r:{std_r}")
print('\n')

data_dir = '../../../MIT_small_train_1'
train_data_dir = data_dir + '/train/'
test_data_dir = data_dir + '/test/'

classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
directions = [train_data_dir + s for s in classes]
directions_test = [test_data_dir + s for s in classes]

mean_b = 0
mean_g = 0
mean_r = 0
std_b = 0
std_g = 0
std_r = 0

it = 0
for dir in directions:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1


for dir in directions_test:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1

mean_b /= it
mean_g /= it
mean_r /= it
std_b /= it
std_g /= it
std_r /= it

print(f"MIT_small_train_1:")
print(f"MEAN:\tb:{mean_b}, g:{mean_g}, r:{mean_r}")
print(f"STD:\tb:{std_b}, g:{std_g}, r:{std_r}")
print('\n')

data_dir = '../../../MIT_small_train_2'
train_data_dir = data_dir + '/train/'
test_data_dir = data_dir + '/test/'

classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
directions = [train_data_dir + s for s in classes]
directions_test = [test_data_dir + s for s in classes]

mean_b = 0
mean_g = 0
mean_r = 0
std_b = 0
std_g = 0
std_r = 0

it = 0
for dir in directions:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1


for dir in directions_test:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1

mean_b /= it
mean_g /= it
mean_r /= it
std_b /= it
std_g /= it
std_r /= it

print(f"MIT_small_train_2:")
print(f"MEAN:\tb:{mean_b}, g:{mean_g}, r:{mean_r}")
print(f"STD:\tb:{std_b}, g:{std_g}, r:{std_r}")
print('\n')

data_dir = '../../../MIT_small_train_3'
train_data_dir = data_dir + '/train/'
test_data_dir = data_dir + '/test/'

classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
directions = [train_data_dir + s for s in classes]
directions_test = [test_data_dir + s for s in classes]

mean_b = 0
mean_g = 0
mean_r = 0
std_b = 0
std_g = 0
std_r = 0

it = 0
for dir in directions:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1


for dir in directions_test:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1

mean_b /= it
mean_g /= it
mean_r /= it
std_b /= it
std_g /= it
std_r /= it

print(f"MIT_small_train_3:")
print(f"MEAN:\tb:{mean_b}, g:{mean_g}, r:{mean_r}")
print(f"STD:\tb:{std_b}, g:{std_g}, r:{std_r}")
print('\n')

data_dir = '../../../MIT_small_train_4'
train_data_dir = data_dir + '/train/'
test_data_dir = data_dir + '/test/'

classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
directions = [train_data_dir + s for s in classes]
directions_test = [test_data_dir + s for s in classes]

mean_b = 0
mean_g = 0
mean_r = 0
std_b = 0
std_g = 0
std_r = 0

it = 0
for dir in directions:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1


for dir in directions_test:
    images = Path(dir).glob('*.jpg')
    for image_path in images:
        img = cv2.imread(image_path.__str__())
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])
        std_b += np.std(img[:, :, 0])
        std_g += np.std(img[:, :, 1])
        std_r += np.std(img[:, :, 2])
        it += 1

mean_b /= it
mean_g /= it
mean_r /= it
std_b /= it
std_g /= it
std_r /= it

print(f"MIT_small_train_4:")
print(f"MEAN:\tb:{mean_b}, g:{mean_g}, r:{mean_r}")
print(f"STD:\tb:{std_b}, g:{std_g}, r:{std_r}")
print('\n')