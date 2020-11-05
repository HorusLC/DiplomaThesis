from tensorflow import keras as krs
# import matplotlib.pyplot as plt
import cv2 as cv

image_size = (256, 256)
batch_size = 8

# def add_augmentation():



def load_validation(path):
    validation_data = krs.preprocessing.image_dataset_from_directory(
        path,
        color_mode='grayscale',
        validation_split=0.2,
        subset='validation',
        seed=1333,
        image_size=image_size,
        batch_size=batch_size,
    )
    return validation_data


def load_dataset(path):
    loaded = krs.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1333,
        validation_split=0.2,
        subset='training'
    )
    return loaded


def load_test_data(path):
    loaded = krs.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1333,
        validation_split=0,
    )
    return loaded


def load_dataset_with_visualization(path):
    loaded = krs.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1333,
        validation_split=0,
    )
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', 600, 600)
    plt.figure(figsize=(10, 10))
    for images, labels in loaded.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            # a= images[i].numpy().astype("uint8")
            cv.imshow('image', images[i].numpy().astype("uint8"))
            cv.waitKey(0)
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.waitforbuttonpress(0)
