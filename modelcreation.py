import tensorflow as tf
from tensorflow import keras as krs
from keras.layers.merge import concatenate
MIN_LEARNING_VALUE = 0.0000001
data_augm = krs.Sequential([
    krs.layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
    krs.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.05, -0.1),
                                                     width_factor=(-0.05, -0.1)),
    krs.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.05, 0.05),
                                                            width_factor=(-0.05, 0.05))])


def model_create_adv_xception(input_shape):
    data_in = krs.Input(shape=input_shape)
    x = data_augm(data_in)
    x = krs.layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = krs.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = krs.layers.BatchNormalization()(x)
    x = krs.layers.Activation("relu")(x)

    x = krs.layers.Conv2D(64, 3, padding="same")(x)
    x = krs.layers.BatchNormalization()(x)
    x = krs.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = krs.layers.Activation("relu")(x)
        x = krs.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = krs.layers.BatchNormalization()(x)

        x = krs.layers.Activation("relu")(x)
        x = krs.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = krs.layers.BatchNormalization()(x)

        x = krs.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = krs.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = krs.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = krs.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = krs.layers.BatchNormalization()(x)
    x = krs.layers.Activation("relu")(x)

    x = krs.layers.GlobalAveragePooling2D()(x)

    x = krs.layers.Dropout(0.6)(x)
    outputs = krs.layers.Dense(1, activation='sigmoid')(x)
    return krs.Model(data_in, outputs)


def model_create_xception(input_shape, num_classes):
    inputs = krs.Input(shape=input_shape)
    # Image augmentation block
    # x = data_augmentation(inputs)

    # Entry block
    x = krs.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = krs.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = krs.layers.BatchNormalization()(x)
    x = krs.layers.Activation("relu")(x)

    x = krs.layers.Conv2D(64, 3, padding="same")(x)
    x = krs.layers.BatchNormalization()(x)
    x = krs.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = krs.layers.Activation("relu")(x)
        x = krs.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = krs.layers.BatchNormalization()(x)

        x = krs.layers.Activation("relu")(x)
        x = krs.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = krs.layers.BatchNormalization()(x)

        x = krs.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = krs.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = krs.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = krs.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = krs.layers.BatchNormalization()(x)
    x = krs.layers.Activation("relu")(x)

    x = krs.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = krs.layers.Dropout(0.5)(x)
    outputs = krs.layers.Dense(units, activation=activation)(x)
    return krs.Model(inputs, outputs)


def lr_scheduler(epoch, lr):
    if epoch < 5 or lr < MIN_LEARNING_VALUE:
        return lr
    elif 5 <= epoch < 15:
        return lr * 0.9
    else:
        return lr * 0.97


def model_create_vgg16(input_shape):
    base_mdl = krs.applications.vgg16.VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    # base_mdl.summary()
    base_mdl.trainable = False
    inputs = krs.Input(input_shape)
    inp = data_augm(inputs)
    inp = krs.applications.vgg16.preprocess_input(inp)
    val = base_mdl(inp, training=False)
    # val = krs.layers.GlobalAveragePooling2D()(val)
    val = krs.layers.Flatten()(val)
    val = krs.layers.Dense(2048, activation="relu")(val)
    val = krs.layers.Dropout(0.1, seed=1333)(val)
    val = krs.layers.Dense(2048, activation="relu")(val)
    # val = krs.layers.Dropout(0.3)(val)
    outputs = krs.layers.Dense(1, activation='sigmoid')(val)
    return krs.Model(inputs, outputs)


def my_inception(layer_input, x1, x2_reduction, x2, x3_reduction, x3, x4):
    # 1x1 conv
    conv1 = krs.layers.Conv2D(x1, (1, 1), padding='same', activation='relu')(layer_input)
    # 3x3 conv
    conv3 = krs.layers.Conv2D(x2_reduction, (1, 1), padding='same', activation='relu')(layer_input)
    conv3 = krs.layers.Conv2D(x2, (3, 3), padding='same', activation='relu')(conv3)
    # 5x5 conv
    conv5 = krs.layers.Conv2D(x3_reduction, (1, 1), padding='same', activation='relu')(layer_input)
    conv5 = krs.layers.Conv2D(x3, (5, 5), padding='same', activation='relu')(conv5)
    # 3x3 max pooling
    pool = krs.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_input)
    pool = krs.layers.Conv2D(x4, (1, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out

def create_inet_full(input_data):
    input_img = krs.layers.Input(input_data)
    val = krs.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(input_img)
    val = krs.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(val)
    val = krs.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(val)
    val = my_inception(val, x1=32, x2_reduction=64, x2=96, x3_reduction=16, x3=32, x4=32)
    val = my_inception(val, x1=64, x2_reduction=96, x2=128, x3_reduction=32, x3=64, x4=32)
    val = my_inception(val, x1=64, x2_reduction=96, x2=128, x3_reduction=32, x3=64, x4=32)
    val = my_inception(val, x1=192, x2_reduction=128, x2=256, x3_reduction=32, x3=96, x4=32)
    val = krs.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(val)
    val = my_inception(val, x1=160, x2_reduction=144, x2=288, x3_reduction=24, x3=48, x4=64)
    val = my_inception(val, x1=160, x2_reduction=144, x2=288, x3_reduction=24, x3=48, x4=64)
    val= krs.layers.MaxPooling2D(pool_size=3,strides=2, padding='same')(val)
    val = my_inception(val, x1=256, x2_reduction=160, x2=320, x3_reduction=32, x3=64, x4=64)
    val = my_inception(val, x1=256, x2_reduction=160, x2=320, x3_reduction=32, x3=64, x4=64)
    val = krs.layers.AveragePooling2D(pool_size=8,strides=1)(val)
    val= krs.layers.Flatten()(val)
    val = krs.layers.Dense(1024, activation='relu')(val)
   # val = krs.layers.Dropout(0.3, seed=1384)(val)
    val = krs.layers.Dense(1024, activation='relu')(val)
    output = krs.layers.Dense(1, activation='sigmoid')(val)
    return krs.Model(input_img, output)


def create_inet_augmented(input_data):
    input_img = krs.layers.Input(input_data)
    val = krs.layers.experimental.preprocessing.Rescaling(1.0 / 255)(input_img)
    val = data_augm(val)
    val = krs.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(val)
    val = krs.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(val)
    val = krs.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(val)
    val = my_inception(val, x1=32, x2_reduction=64, x2=96, x3_reduction=16, x3=32, x4=32)
    val = my_inception(val, x1=64, x2_reduction=96, x2=128, x3_reduction=32, x3=64, x4=32)
    val = my_inception(val, x1=64, x2_reduction=96, x2=128, x3_reduction=32, x3=64, x4=32)
    val = my_inception(val, x1=192, x2_reduction=128, x2=256, x3_reduction=32, x3=96, x4=32)
    val = krs.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(val)
    val = my_inception(val, x1=160, x2_reduction=144, x2=288, x3_reduction=24, x3=48, x4=64)
    val = my_inception(val, x1=160, x2_reduction=144, x2=288, x3_reduction=24, x3=48, x4=64)
    val = krs.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(val)
    val = my_inception(val, x1=256, x2_reduction=160, x2=320, x3_reduction=32, x3=64, x4=64)
    val = my_inception(val, x1=256, x2_reduction=160, x2=320, x3_reduction=32, x3=64, x4=64)
    val = krs.layers.AveragePooling2D(pool_size=8, strides=1)(val)
    val = krs.layers.Flatten()(val)
    val = krs.layers.Dense(1024, activation='relu')(val)
    val = krs.layers.Dropout(0.2, seed=1384)(val)
    val = krs.layers.Dense(512, activation='relu')(val)
    output = krs.layers.Dense(1, activation='sigmoid')(val)
    return krs.Model(input_img, output)


def create_inet(input_data):
    input_img = krs.layers.Input(input_data)
    val = krs.layers.experimental.preprocessing.Rescaling(1.0/255)(input_img)
    val = data_augm(val)
    val = krs.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(val)
    val = krs.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(val)
    val = krs.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(val)
    val = my_inception(val, x1=32, x2_reduction=64, x2=96, x3_reduction=16, x3=32, x4=32)
    val = my_inception(val, x1=64, x2_reduction=96, x2=128, x3_reduction=32, x3=64, x4=32)
    val = my_inception(val, x1=64, x2_reduction=96, x2=128, x3_reduction=32, x3=64, x4=32)
    val = my_inception(val, x1=192, x2_reduction=128, x2=256, x3_reduction=32, x3=96, x4=32)
    val = krs.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(val)
    val = my_inception(val, x1=160, x2_reduction=144, x2=288, x3_reduction=24, x3=48, x4=64)
    val = my_inception(val, x1=160, x2_reduction=144, x2=288, x3_reduction=24, x3=48, x4=64)
    val= krs.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(val)
    val = my_inception(val, x1=256, x2_reduction=160, x2=320, x3_reduction=32, x3=64, x4=64)
    val = my_inception(val, x1=256, x2_reduction=160, x2=320, x3_reduction=32, x3=64, x4=64)
    val = krs.layers.AveragePooling2D(pool_size=8, strides=1)(val)
    val= krs.layers.Flatten()(val)
    val = krs.layers.Dense(2048, activation='relu')(val)
    val = krs.layers.Dropout(0.2, seed=1384)(val)
    val = krs.layers.Dense(2048, activation='relu')(val)
    output = krs.layers.Dense(1, activation='sigmoid')(val)
    return krs.Model(input_img, output)
