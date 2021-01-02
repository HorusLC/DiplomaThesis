import datamgmt as datalib
import modelcreation as mdl
from tensorflow import keras as keras
import datetime as dt
import time as ti
import pickle as pickle
import resultsplotter as resplot
import json as json
import matplotlib as plotter
import pandas as panda
from keras.callbacks import TensorBoard
import inspect as inspect_tool
import tensorflow as tf

# TRAINING_PATH = 'E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/train' old paths
# TEST_PATH = 'E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/test' old paths
TRAINING_PATH = 'E:/pycharmProjects/DiplomaThesis/dataset/train'
TEST_PATH = 'E:/pycharmProjects/DiplomaThesis/dataset/test'
VAL_PATH = 'E:/pycharmProjects/DiplomaThesis/dataset/val'
tf.random.set_seed(1304)


def train_vgg(experiment_folder, learning_rate, num_epochs):
    identfier = experiment_folder + ti.strftime('-%Y-%m-%d-%H%M%S')
    tensorboard = TensorBoard(log_dir='logs/{}'.format(identfier))
    training = datalib.load_dataset(TRAINING_PATH)
    val_ds = datalib.load_dataset(VAL_PATH)
    model = mdl.model_create_vgg16(input_shape=(128, 128, 3))
    model.summary()
    folder_path = datalib.init_model_folder(dir_name=experiment_folder,
                                            optimizer='sgd',
                                            lr=str(learning_rate),
                                            image_dimensions='128,128,3',
                                            architecture=inspect_tool.getsource(
                                                mdl.model_create_vgg16))
    callbacks = [
        keras.callbacks.ModelCheckpoint(folder_path + "/save_at_{epoch}.h5", ),
        tensorboard,
        # keras.callbacks.LearningRateScheduler(mdl.lr_scheduler, verbose=1)
        # keras.callbacks.CSVLogger(filename="xception_log.csv", separator=',', append=True)
        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=0.005, mode='auto')
    ]
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC",
                 keras.metrics.TruePositives(),
                 keras.metrics.FalsePositives(),
                 keras.metrics.TrueNegatives(),
                 keras.metrics.FalseNegatives()]
    )
    history = model.fit(
        training, epochs=num_epochs, callbacks=callbacks, validation_data=val_ds,
    )
    # dump results binary and json
    with open(folder_path + '/adam_history', mode='wb') as history_file:
        pickle.dump(history.history, history_file)
    with open(folder_path + '/adam_hist.json', mode='w') as file:
        dataframe = panda.DataFrame(history.history)
        dataframe.to_json(file, indent=4)

    # plot after all epochs
    datalib.plot_history_accuracy(history, folder_path)
    datalib.plot_history_loss_func(history, folder_path)


def train_model_on_data():
    training = datalib.load_dataset(TRAINING_PATH)
    val_ds = datalib.load_validation(TRAINING_PATH)
    model = mdl.model_create_adv_xception(input_shape=datalib.image_size + (1,))
    epochs = 20

    # keras.backend.clear_session()
    callbacks = [
        keras.callbacks.ModelCheckpoint("128/save_at_{epoch}.h5"),
        # keras.callbacks.CSVLogger(filename="xception_log.csv", separator=',', append=True)
    ]
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC",
                 keras.metrics.TruePositives(),
                 keras.metrics.FalsePositives(),
                 keras.metrics.TrueNegatives(),
                 keras.metrics.FalseNegatives()]
    )
    model.summary()
    history = model.fit(
        training, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )
    # dump results binary andjson
    with open('128/rms_history', mode='wb') as history_file:
        pickle.dump(history.history, history_file)
    with open('128/rms_hist.json', mode='w') as file:
        dataframe = panda.DataFrame(history.history)
        dataframe.to_json(file)

    # plot after all epochs
    datalib.plot_history_accuracy(history)
    datalib.plot_history_loss_func(history)


def eval_single_model(test_data, path):
    mod = keras.models.load_model(path)
    mod.summary()
    print(str(mod.evaluate(test_data, verbose=1)))


def eval_model(test_data):
    output_file = str(ti.strftime('%Y-%m-%d-%H%M%S')) + '.txt'
    with open(output_file, mode='a') as f:
        for i in range(36, 46, 1):  # from 1 to 8(not included) with a step of 1
            path = 'save_ep_' + str(i) + '.h5'
            mod = keras.models.load_model(path)
            print(
                str(dt.datetime.now()) + ' results for save-' + str(i) + ' are ---> '
                + str(mod.evaluate(test_data, verbose=1)[1] * 100), file=f)
    f.close()
    # results = mod.evaluate(test_data, verbose=1)


def continue_training(from_epoch, to_epoch, loading_path, saving_path, hist_path):
    trn = datalib.load_dataset(TRAINING_PATH)
    val = datalib.load_dataset(VAL_PATH)
    mod = keras.models.load_model(loading_path)
    clbcks = [
        keras.callbacks.ModelCheckpoint(saving_path),
        # keras.callbacks.CSVLogger(filename="xception_log.csv", separator=',', append=True)

    ]
    hist = mod.fit(trn, initial_epoch=from_epoch, epochs=to_epoch, callbacks=clbcks, validation_data=val)
    with open(hist_path, mode='r') as readingfile:
        hist_datafr = panda.read_json(readingfile)
        print(hist_datafr)
        new_datafr = panda.DataFrame(hist.history)
        updated_datafr = hist_datafr.append(new_datafr, ignore_index=True)
    with open(hist_path, mode='w') as writingfile:
        updated_datafr.to_json(writingfile)


def train_inception(experiment_folder, learning_rate, num_epochs):
    identfier = experiment_folder + ti.strftime('-%Y-%m-%d-%H%M%S')
    tensorboard = TensorBoard(log_dir='logs/{}'.format(identfier))
    training = datalib.load_dataset(TRAINING_PATH)
    val_ds = datalib.load_dataset(VAL_PATH)
    i_model = mdl.create_inet((128, 128, 1))
    i_model.summary()
    folder_path = datalib.init_model_folder(dir_name=experiment_folder,
                                            optimizer='adam',
                                            lr=str(learning_rate),
                                            image_dimensions='128,128,1',
                                            architecture=inspect_tool.getsource(
                                                mdl.create_inet))
    callbacks = [
        keras.callbacks.ModelCheckpoint(folder_path + "/save_at_{epoch}.h5", ),
        tensorboard,
        # keras.callbacks.LearningRateScheduler(mdl.lr_scheduler, verbose=1)
        # keras.callbacks.CSVLogger(filename="xception_log.csv", separator=',', append=True)
        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=0.005, mode='auto')
    ]
    i_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC",
                 keras.metrics.TruePositives(),
                 keras.metrics.FalsePositives(),
                 keras.metrics.TrueNegatives(),
                 keras.metrics.FalseNegatives()]
    )
    history = i_model.fit(
        training, epochs=num_epochs, callbacks=callbacks, validation_data=val_ds,
    )
    # dump results binary and json
    with open(folder_path + '/adam_history', mode='wb') as history_file:
        pickle.dump(history.history, history_file)
    with open(folder_path + '/adam_hist.json', mode='w') as file:
        dataframe = panda.DataFrame(history.history)
        dataframe.to_json(file, indent=4)
    datalib.plot_history_accuracy(history)
    datalib.plot_history_loss_func(history)


if __name__ == '__main__':
    keras.backend.clear_session()
    #train_inception('inception_v2', 0.0001, 30)
    #train_vgg('vgg_test', 0.000001, 60)
    # test_data = datalib.load_dataset(TEST_PATH)
    # eval_single_model(test_data, path='new_ds/vgg_initial/save_at_3.h5')
    resplot.plot_history('new_ds/inception_v2/adam_hist.json', max_x=31, interval=2)
    # continue_training(60, 90, 'new_ds/vgg_v8reg2/save_at_60.h5', 'new_ds/vgg_v8reg2/save_at_{epoch}.h5',
    #                   hist_path='new_ds/vgg_v8reg2/adam_hist.json')
    # train_vgg()
    # datalib.load_dataset_with_visualization(TRAINING_PATH)
    # datalib.split_data(input_data='E:/pycharmProjects/DiplomaThesis/dataset2/train',
    #                  data_ratio=(0.8, 0.2))
#  test_data = datalib.load_test_data(TEST_PATH)
# train_dat = datalib.load_dataset(TRAINING_PATH)
# print('hello')
# mod = mdl.model_create_vgg16(input_shape=(128, 128, 3))
# mod.summary()
# eval_single_model(test_data, 'vggtransfer512/save_at_15.h5')
# train_vgg()
# continue_training(from_epoch=100,
#                   to_epoch=130,
#                   saving_path='vgg/save_at_{epoch}.h5',
#                   loading_path='vgg/save_at_100.h5',
#                   hist_path='vgg/rms_hist.json')
#
# resplot.plot_history('new_ds/vgg_2/adam_hist.json',interval=1, max_x=30)

# train_model_on_data()
# t_data = datalib.load_test_data(TEST_PATH)
# eval_single_model(t_data, path='xception_2_rmsprop/save_at_40.h5')
# eval_model(t_data)
# training = datalib.load_dataset(TRAINING_PATH)
# val_ds = datalib.load_validation(TRAINING_PATH)
# model = keras.models.load_model("128/save_at_20.h5")
# callbacks = [
#      keras.callbacks.ModelCheckpoint("128/save_at_{epoch}.h5"),
#      # keras.callbacks.CSVLogger(filename="xception_log.csv", separator=',', append=True)
#
#  ]
# history = model.fit(training, initial_epoch=20, epochs=45, callbacks=callbacks, validation_data=val_ds)
# with open('128/rms_hist.json', mode='r') as file:
#     hist_df = panda.read_json(file)
#     print(hist_df)
#     hist_df.plot(use_index=True, y=['loss', 'val_loss'], xticks=range(0, 45, 1))
#     plotter.pyplot.show()
#     datalib.plot_dict_accuracy(hist_df.to_dict())
#     datalib.plot_dict_loss_func(hist_df.to_dict())
#     new_df = panda.DataFrame(history.history)
#     updated_df = hist_df.append(new_df, ignore_index=True)
# with open('128/rms_hist.json', mode='w') as f:
#     updated_df.to_json(f)
#  print(model.optimizer.learning_rate)
# history = model.fit(training, initial_epoch=65, epochs=75, callbacks=callbacks, validation_data=val_ds)
# history = panda.read_csv('xception_log.csv', sep=',', engine='python')
# region plotting from history file
# datalib.plot_history_accuracy(history)
# datalib.plot_history_loss_func(history)
# endregion
