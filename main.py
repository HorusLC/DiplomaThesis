import datamgmt as datalib
import modelcreation as mdl
from tensorflow import keras as keras
import datetime as dt
import time as ti

TRAINING_PATH = 'E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/train'
TEST_PATH = 'E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/test'


def train_model_on_data():
    training = datalib.load_dataset(TRAINING_PATH)
    val_ds = datalib.load_validation('E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/train')
    model = mdl.model_create_xception(input_shape=datalib.image_size + (1,), num_classes=2)
    n = datalib.image_size + (3,)
    print(n)
    epochs = 7
    # keras.backend.clear_session()
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        training, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )


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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t_data = datalib.load_test_data(TEST_PATH)
    eval_model(t_data)
    # training = datalib.load_dataset(TRAINING_PATH)
    # val_ds = datalib.load_validation('E:/kaggle/chest_xray_imgs_pneumonia/archive/chest_xray/train')
    # model = keras.models.load_model("save_ep_35.h5")
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("save_ep_{epoch}.h5"),
    # ]
    #
    # model.fit(training, initial_epoch=35, epochs=45, callbacks=callbacks, validation_data=val_ds)
