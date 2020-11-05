
from tensorflow import keras as krs


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


def model_create_vgg16(input_shape):
    inputs = krs.layers.Input(shape=input_shape)
    mdl = krs.applications.vgg16.VGG16(include_top=False, input_tensor=inputs, weights=None)
    x = mdl.output

    x = krs.layers.Activation('relu')(x)
    x = krs.layers.GlobalAveragePooling2D()(x)
    x = krs.layers.Flatten()(x)
    output = krs.layers.Dense(1, activation='sigmoid')(x)

    return krs.Model(inputs, output)
