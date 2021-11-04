import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.compat.v1.logging import set_verbosity, ERROR
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    InputLayer,
    PReLU,
    Input,
    MaxPooling2D)

from tensorflow.keras.initializers import HeNormal, RandomNormal
from tensorflow.keras.applications import VGG16
from image_config import (
    HR_IMG_SIZE,
    LR_IMG_SIZE,
    INIT_TILE_SIZE,
    PADDING,
    UPSCALING_FACTOR,
    COLOR_CHANNELS
)


def fsrcnn(
        d: int,
        s: int,
        m: int,
        tiled: bool,
        upscaling_factor: int = UPSCALING_FACTOR,
        color_channels: int = COLOR_CHANNELS):
    """
    Generate and build the FSRCNN model.

    :param d: fsrcnn d parameter.
    :param s: fsrcnn s parameter
    :param m: fsrcnn m parameter
    :param tiled: tiled input or not? in bool.
    :param upscaling_factor: final upscaling factor
    :param color_channels: ndim image channels.
    :returns a fsrcnn model.
    """

    set_verbosity(ERROR)

    LR_TILE_SIZE = (
        PADDING + INIT_TILE_SIZE[0],
        PADDING + INIT_TILE_SIZE[1])

    HR_TILE_SIZE = (
        (PADDING + LR_TILE_SIZE[0]) * UPSCALING_FACTOR, 
        (PADDING + LR_TILE_SIZE[1]) * UPSCALING_FACTOR)

    if tiled:
        input_size = LR_TILE_SIZE
    else:
        input_size = LR_IMG_SIZE

    model = Sequential()
    model.add(InputLayer(input_shape=(input_size[0],
                                      input_size[1],
                                      color_channels)))

    # feature extraction
    model.add(Conv2D(
        kernel_size=5,
        filters=d,
        padding="same",
        kernel_initializer=HeNormal(), ))
    model.add(PReLU(alpha_initializer="zeros", shared_axes=[1, 2]))

    # shrinking
    model.add(Conv2D(
        kernel_size=1,
        filters=s,
        padding="same",
        kernel_initializer=HeNormal()))
    model.add(PReLU(alpha_initializer="zeros", shared_axes=[1, 2]))

    # mapping
    for _ in range(m):
        model.add(Conv2D(
            kernel_size=3,
            filters=s,
            padding="same",
            kernel_initializer=HeNormal(), ))
        model.add(PReLU(alpha_initializer="zeros", shared_axes=[1, 2]))

    # expanding
    model.add(Conv2D(
        kernel_size=1,
        filters=d,
        padding="same"))
    model.add(PReLU(alpha_initializer="zeros", shared_axes=[1, 2]))

    # deconvolution
    model.add(Conv2DTranspose(
        kernel_size=9,
        filters=color_channels,
        strides=upscaling_factor,
        padding="same",
        kernel_initializer=RandomNormal(mean=0, stddev=0.001)))
    return model


def vgg_block(inputs, f: int, d: int, m: int, block_name: str):
    set_verbosity(ERROR)

    x = inputs

    for repetition in range(m):
        x = Conv2D(
            f,
            (d, d),
            activation='relu',
            padding='same',
            name=block_name + "_conv" + str(repetition + 1)
        )(x)

    x = MaxPooling2D(
        (2, 2),
        strides=(2, 2),
        name=block_name + "_pool")(x)

    return x


def vgg_loss(block_nums: list, tiled: bool):
    """

    :param block_nums: tuple of indexes choosen VGG Layer
    :param tiled: bool type of if input tiled or not.
    :return: tuples of keras loss network models
    """
    set_verbosity(ERROR)

    LR_TILE_SIZE = (
        PADDING + INIT_TILE_SIZE[0],
        PADDING + INIT_TILE_SIZE[1])

    HR_TILE_SIZE = (
        (PADDING + LR_TILE_SIZE[0]) * UPSCALING_FACTOR, 
        (PADDING + LR_TILE_SIZE[1]) * UPSCALING_FACTOR)

    if tiled:
        img_shape=(HR_TILE_SIZE[0], HR_TILE_SIZE[1], 3)
    else:
        img_shape=(HR_IMG_SIZE[0], HR_IMG_SIZE[1], 3)

    weights = "utils/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    __vgg= VGG16(
        include_top=False, 
        weights=weights,
        input_shape=img_shape
    )
    __vgg.trainable = False
    loss_models = []

    for layer in block_nums:
        loss_models.append(
            Model(
                inputs=__vgg.input,
                outputs=__vgg.layers[layer].output,
                name="VGGlossNetwork_layer" + str(layer)
            ))

    for x in loss_models:
        x.trainable = False

    del __vgg
    return loss_models
