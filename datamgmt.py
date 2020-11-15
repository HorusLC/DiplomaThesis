from tensorflow import keras as krs
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.pyplot as plotter
import numpy
image_size = (128, 128) #180
batch_size = 32

# def add_augmentation():


def plot_dict_accuracy(history):                      # uses dictionary as input
    plotter.plot(history['accuracy'])
    plotter.plot(history['val_accuracy'])
    plotter.title('accuracy of model advanced xception')
    plotter.ylabel('accuracy')
    plotter.xlabel('epoch')
    plotter.legend(['train', 'val'], loc='upper left')
    plotter.savefig('history_acc.png')
    plotter.show()


def plot_history_accuracy(history):                    #uses history object as input (returned by model.fit())
    plotter.plot(history.history['accuracy'])
    plotter.plot(history.history['val_accuracy'])
    plotter.title('accuracy of model advanced xception')
    plotter.ylabel('accuracy')
    plotter.xlabel('epoch')
    plotter.legend(['train', 'val'], loc='upper left')
    plotter.savefig('history_acc.png')
    plotter.show()


def plot_dict_loss_func_(history):
    plotter.plot(history['loss'])
    plotter.plot(history['val_loss'])
    plotter.title('model loss')
    plotter.ylabel('loss')
    plotter.xlabel('epoch')
    plotter.legend(['train', 'val'], loc='upper left')
    plotter.savefig('history_lossval.png')
    plotter.show()


def plot_history_loss_func(history):
    plotter.plot(history.history['loss'])
    plotter.plot(history.history['val_loss'])
    plotter.title('model loss')
    plotter.ylabel('loss')
    plotter.xlabel('epoch')
    plotter.legend(['train', 'val'], loc='upper left')
    plotter.savefig('history_lossval.png')
    plotter.show()


def load_validation(path):
    validation_data = krs.preprocessing.image_dataset_from_directory(
        path,
        color_mode='rgb',
        validation_split=0.2,
        subset='validation',
        seed=1333,
        labels='inferred',
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    return validation_data


def load_dataset(path):
    loaded = krs.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1333,
        validation_split=0.2,
        subset='training',
        label_mode='binary'
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
        label_mode='binary'
    )
    return loaded


def load_viz(path):
    validation_data = krs.preprocessing.image_dataset_from_directory(
        path,
        color_mode='grayscale',
        validation_split=0.2,
        subset='validation',
        seed=1333,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    plt.figure(figsize=(10, 10))
    for images, labels in validation_data.take(1):
        for i in range(8):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")



def load_dataset_with_visualization(path):
    loaded = krs.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        color_mode='grayscale',
        subset="validation",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1333,
        validation_split=0.2,
    )
    #cv.namedWindow('image', cv.WINDOW_NORMAL)
    #cv.resizeWindow('image', 600, 600)
    plt.figure(figsize=(10, 10))
    for images, labels in loaded.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            # a= images[i].numpy().astype("uint8")
            #cv.imshow('image', images[i].numpy().astype("uint8"))
            #cv.waitKey(0)
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.waitforbuttonpress(0)
