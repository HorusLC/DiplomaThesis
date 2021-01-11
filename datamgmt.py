from tensorflow import keras as krs
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.pyplot as plotter
import numpy
import splitfolders as data_splitter
import os as os
import time as ti
import sys
import modelcreation as models
import random as random

image_size = (128, 128)  # 180
batch_size = 32

# def add_augmentation():

path1 = 'E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/train'


def init_model_folder(dir_name, optimizer, lr, architecture, image_dimensions):
    folder_path = 'new_ds/{}'.format(dir_name)
    os.mkdir(path=folder_path)
    info_file = folder_path + '/' + str(ti.strftime('%H-%M-%d-%m-%y')) + '.txt'
    with open(info_file, mode='a') as f:
        print(str(f'optimizer: {optimizer}\n'), file=f)
        print(str(f'learning rate: {lr}\n'), file=f)
        print(str(f'image dimensions used: {image_dimensions}'), file=f)
        print(str(f'architecture is:\n{architecture}'), file=f)

    return folder_path


def plot_dict_accuracy(history):  # uses dictionary as input
    plotter.plot(history['accuracy'])
    plotter.plot(history['val_accuracy'])
    plotter.title('accuracy of model advanced xception')
    plotter.ylabel('accuracy')
    plotter.xlabel('epoch')
    plotter.legend(['train', 'val'], loc='upper left')
    plotter.savefig('history_acc.png')
    plotter.show()


def split_data(input_data, output_folder, data_ratio):
    if not os.path.exists(input_data):
        print('Folder for input data not found!')
        return
    folder_exist = os.path.exists(output_folder)
    if folder_exist:
        files = os.listdir(output_folder)
        if len(files) > 0:
            print('Warning! Folder already exists and contains other information!\n'
                  'Please pass a name/path for a new folder')
            return
    try:
        data_splitter.ratio(
            input=input_data,
            output=output_folder,
            seed=1384,
            ratio=data_ratio)

    except BlockingIOError as bioe:
        print('IO operation has been blocked by other process already using :' + bioe.filename)
    except MemoryError as mem:
        print('Insufficient memory: ' + str(mem))
    except PermissionError as pe:
        print('You dont have the permission to perform this operation')
    except IOError as ioe:
        print('An IO error occured, try again!')


def plot_history_accuracy(history, path):  # uses history object as input (returned by model.fit())
    plotter.plot(history.history['accuracy'])
    plotter.plot(history.history['val_accuracy'])
    plotter.title('accuracy of model')
    plotter.ylabel('accuracy')
    plotter.xlabel('epoch')
    plotter.legend(['train', 'val'], loc='upper left')
    plotter.savefig(path + '/history_acc.png')
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


def plot_history_loss_func(history, path):
    plotter.plot(history.history['loss'])
    plotter.plot(history.history['val_loss'])
    plotter.title('model loss')
    plotter.ylabel('loss')
    plotter.xlabel('epoch')
    plotter.legend(['train', 'val'], loc='upper left')
    plotter.savefig(path + '/history_lossval.png')
    plotter.show()


def load_validation(path):
    validation_data = krs.preprocessing.image_dataset_from_directory(
        path,
        color_mode='rgb',
        seed=1384,
        labels='inferred',
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    return validation_data


def show_files(folder_path):
    if not os.path.exists(folder_path):
        print('The path: ' + folder_path + 'was not found')
        return
    folder = os.listdir(folder_path)
    random.shuffle(folder)
    for file in folder:
        print(file)


# noinspection PyBroadException
def count_class_files(path_normal, path_pneumonia):
    normal_count = 0
    pneumonia_count = 0
    if not os.path.exists(path_normal):
        print('The path: ' + path_normal + 'was not found')
        return
    if not os.path.exists(path_pneumonia):
        print('The path: ' + path_pneumonia + ' was not found')
        return
    try:
        for file_n in os.listdir(path_normal):
            if os.path.isfile(os.path.join(path_normal, file_n)):
                normal_count += 1

        for file_p in os.listdir(path_pneumonia):
            if os.path.isfile(os.path.join(path_pneumonia, file_p)):
                pneumonia_count += 1

        labels = 'NORMAL', 'PNEUMONIA'
        counts = [normal_count, pneumonia_count]
        pie_fig, pie_axis = plotter.subplots()
        pie_axis.pie(counts, labels=labels, explode=(0, 0.05), shadow=True, startangle=80,
                     autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(counts) / 100))
        plotter.show()
    except FileNotFoundError as fnf:
        print('The file was not found! Check if ' + fnf.filename + 'is correct')
    except NotADirectoryError as nde:
        print('Specified path: ' + nde.filename + 'was not a directory')
    except Exception:
        print('Something went wrong: ' + str(sys.exc_info()[0]))


def visualize_partition():
    fig, axis = plotter.subplots()
    axis.pie((80, 10, 10), labels=('train', 'test', 'validation'), autopct='%.2f%%', explode=(0, 0.05, 0.05),
             startangle=90)
    plotter.show()


def calculate_precision_sensitivity(tp,fp,tn,fn):
    precision = tp/(tp+fp)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    f1 = 2*tp/(2*tp+fp+fn)
    print(f'specificity on test data is {specificity}')
    print(f'precision on test data is {precision}')
    print(f'sensitivity on test data is {sensitivity}')
    print(f'f1 score is {f1}')


def load_dataset(path):
    loaded = krs.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1384,
        label_mode='binary'
    )
    return loaded


def load_test_data(path):
    loaded = krs.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1384,
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
        validation_split=0.1,
    )
    plt.figure(figsize=(10, 10))
    for images, labels in loaded.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'), cmap='gray', vmin=0, vmax=255)
            if labels[i] == 1:
                title = 'Pneumonia'
            else:
                title = 'Normal'
            plt.title(title)
            plt.axis("off")
    plt.show()
