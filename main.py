import datamgmt as datalib
import modelcreation as mdl
from tensorflow import keras as keras
import datetime as dt
import time as ti
import pickle as pickle
import json as json
import matplotlib as plotter
import pandas as panda

TRAINING_PATH = 'E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/train'
TEST_PATH = 'E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/test'


def train_model_on_data():
    training = datalib.load_dataset(TRAINING_PATH)
    val_ds = datalib.load_validation(TRAINING_PATH)
    model = mdl.model_create_adv_xception(input_shape=datalib.image_size + (1,))
    epochs = 45

    # keras.backend.clear_session()
    callbacks = [
        keras.callbacks.ModelCheckpoint("xc2_rms_nds/save_at_{epoch}.h5"),
        # keras.callbacks.CSVLogger(filename="xception_log.csv", separator=',', append=True)
    ]
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC",
                 keras.metrics.TruePositives(),
                 keras.metrics.FalsePositives(),
                 keras.metrics.TrueNegatives(),
                 keras.metrics.FalseNegatives()]
    )

    history = model.fit(
        training, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )
    #dump results binary andjson
    with open('xc2_rms_nds/rms_history', mode='wb') as history_file:
        pickle.dump(history.history, history_file)
    with open('xc2_rms_nds/rms_hist.json', mode='w') as file:
        dataframe = panda.DataFrame(history.history)
        dataframe.to_json(file)

    #plot after all epochs
    datalib.plot_history_accuracy(history)
    datalib.plot_history_loss_func(history)

def eval_single_model(test_data,path):
    mod = keras.models.load_model(path)
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


if __name__ == '__main__':
    #val= datalib.load_validation(TRAINING_PATH)
    a=datalib.load_viz(TRAINING_PATH)
    print('a')
    # datalib.load_dataset_with_visualization(TRAINING_PATH)
    #train_model_on_data()
    #t_data = datalib.load_test_data(TEST_PATH)
    # eval_single_model(t_data, path='xception_2_rmsprop/save_at_40.h5')
    # eval_model(t_data)
    # training = datalib.load_dataset(TRAINING_PATH)
    # val_ds = datalib.load_validation('E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/train')
    # model = keras.models.load_model("xception_2_rmsprop/save_at_21.h5")
    # callbacks = [
    #      keras.callbacks.ModelCheckpoint("xception_2_rmsprop/save_at_{epoch}.h5"),
    #      # keras.callbacks.CSVLogger(filename="xception_log.csv", separator=',', append=True)
    #
    #  ]
    # history = model.fit(training, initial_epoch=21, epochs=45, callbacks=callbacks, validation_data=val_ds)
    # with open('xception_2_rmsprop/rms_hist.json', mode='r') as file:
    #     hist_df = panda.read_json(file)
    #     print(hist_df)
    #     hist_df.plot(use_index=True, y=['loss', 'val_loss'], xticks=range(1,45,1))
    #     plotter.pyplot.show()
       # datalib.plot_dict_accuracy(hist_df.to_dict())
       #datalib.plot_dict_loss_func(hist_df.to_dict())
    #     new_df = panda.DataFrame(history.history)
    #     updated_df = hist_df.append(new_df, ignore_index=True)
    # with open('xception_2_rmsprop/rms_hist.json', mode='w') as f:
    #     updated_df.to_json(f)
#  print(model.optimizer.learning_rate)
# history = model.fit(training, initial_epoch=65, epochs=75, callbacks=callbacks, validation_data=val_ds)
# history = panda.read_csv('xception_log.csv', sep=',', engine='python')
# datalib.plot_history_accuracy(history)
# datalib.plot_history_loss_func(history)
