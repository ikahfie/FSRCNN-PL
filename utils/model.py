from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    InputLayer,
    PReLU,
    Input,
    MaxPooling2D)
from tensorflow.keras.initializers import HeNormal, RandomNormal
try:
    from utils.image_config import (
        HR_IMG_SIZE,
        LR_IMG_SIZE,
        UPSCALING_FACTOR,
        COLOR_CHANNELS)
except ModuleNotFoundError:
    from image_config import (
        HR_IMG_SIZE,
        LR_IMG_SIZE,
        UPSCALING_FACTOR,
        COLOR_CHANNELS)


def create_model(
        d: int,
        s: int,
        m: int,
        input_size: tuple = LR_IMG_SIZE,
        upscaling_factor: int = UPSCALING_FACTOR,
        color_channels: int = COLOR_CHANNELS):
        
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


def vgg_block(input, f:int, d:int, m:int, block_name:str):
    x = input 

    for repetition in range(m):
        x = Conv2D(
            f, 
            (d, d), 
            activation='relu', 
            padding='same', 
            name=block_name+"_conv"+ str(repetition+1)
            )(x)

    x = MaxPooling2D(
        (2, 2), 
        strides=(2, 2), 
        name=block_name+"_pool")(x)

    return x

def vgg_loss(block_nums:list):

    img_input = Input(shape=(HR_IMG_SIZE[0], HR_IMG_SIZE[1], 3))
    # Block 1
    x = vgg_block(img_input, 64, 3, 2, "block1")
    # Block 2
    x = vgg_block(x, 128, 3, 2, "block2")
    # Block 3
    x = vgg_block(x, 256, 3, 3, "block3")
    # Block 4
    x = vgg_block(x, 512, 3, 3, "block4")
    # Block 5
    x = vgg_block(x, 512, 3, 3, "block5")

    __vgg = Model(inputs=img_input, outputs=x)
    try:
        __vgg.load_weights("utils/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    except OSError:
        __vgg.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

    __vgg.trainable = False
    # return __vgg

    loss_models=[]
    for layer in block_nums:
        loss_models.append(
            Model(
                inputs=__vgg.input, 
                outputs=__vgg.layers[layer].output,
                name="VGGlossNetwork_layer"+str(layer)
                ))

    for x in loss_models:
        x.trainable=False

    del (__vgg)
    return loss_models


def build_fsrcnn_with_vgg_loss(config: dict):
    
    model = create_model(
        config["d"],config["s"],config["m"],
        (224//2, 224//2), 2, 3)
    
    out = []
    loss_models = vgg_loss(config["vgg_block_nums"])

    for models in loss_models:
        out.append(models(model.output))


    return Model(inputs=model.input, outputs=out,name="fsrcnn_vggloss")


if __name__ == "__main__":
    config = {
        "dataset_dir": "../input/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR/",
        "k": 2,
        "d": 32,
        "s": 5,
        "m": 1,
        "lr": 0.001,
        "epoch": 500,
        "batch_size": 8,
        "steps_per_epoch": 4,
        "val_batch_size": 8,
        "val_steps": 10,
        "weight_path": "weights/model_{epoch:05d}.h5",
        "vgg_block_nums": [1, 2, 5, 9]
    }

    model = build_fsrcnn_with_vgg_loss(config)
    model.summary()
