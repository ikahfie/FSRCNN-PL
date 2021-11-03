import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from utils.dataloader import DIV2K
except ModuleNotFoundError:
    from dataloader import DIV2K

try:
    from utils.lossAndMetrics import perceptual_loss, metric_psnr, metric_ssim
except ModuleNotFoundError:
    from lossAndMetrics import perceptual_loss, metric_psnr, metric_ssim

try:
    from utils.model import fsrcnn
except ModuleNotFoundError:
    from model import fsrcnn

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint)


def train(config: dict) -> None:
    """
    Main train pipeline function. generate input, training model,
    based on the config dict params.

    :param config: main config file
    :return: None
    """
    print(
        " ####################################################################\n",
        "*----------------------------   CONFIG   --------------------------*\n",
        "####################################################################\n",
    )

    for x in config:
        print("{}: {}".format(x, config[x]))
    print()

    d = config["d"]
    s = config["s"]
    m = config["m"]

    train_dataset = DIV2K(
        hr_image_folder=config["train_dir"],
        batch_size=config["batch_size"],
        set_type="",
        tiled=config["tiled"]
    )

    val_dataset = DIV2K(
        hr_image_folder=config["val_dir"],
        batch_size=config["val_batch_size"],
        set_type="VAL",
        tiled=config["tiled"]
    )

    model = fsrcnn(
        d=d, s=s, m=m,
        tiled=config["tiled"]
    )

    if config["pretrainedWeightPath"] != "":
        model.load_weights(config["pretrainedWeightPath"])
        model.trainable = False
        model.layers[-1].trainable = True

    lr = config["lr"]

    model.add_loss(
        perceptual_loss(
            alpha=config["alpha"],
            vgg_layer_nums=config["vgg_layer_nums"],
            tiled=config["tiled"]
        )
    )

    model.add_metrics(
        ["accuracy", metric_psnr, metric_ssim]
    )

    model.compile(
        optimizer=RMSprop(learning_rate=lr),
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-6,
        patience=50,
        verbose=1,
        restore_best_weights=True,
    )

    ckpt = ModelCheckpoint(
        filepath=config["output_weight_path"],
        monitor='val_loss',
        save_best_only=True,
        save_freq="epoch",
        save_weights_only=False,
        verbose=0
    )

    print(
        " ####################################################################\n",
        "*------------------------   FITTING MODEL   -----------------------*\n",
        "####################################################################\n",
    )

    history = model.fit(
        train_dataset,
        epochs=config["epoch"],
        # steps_per_epoch=config["steps_per_epoch"],
        callbacks=[reduce_lr, early_stop, ckpt],
        validation_data=val_dataset,
        # validation_steps=config["val_steps"],
        verbose=0
    )


if __name__ == "__main__":
    training_config = {
        "train_dir": "dataset/DIV2K_train_HR",
        "val_dir": "dataset/DIV2K_valid_HR",
        "d": 32,
        "s": 5,
        "m": 1,
        "lr": 0.001,
        "epoch": 500,
        "batch_size": 32,
        "steps_per_epoch": 4,
        "val_batch_size": 32,
        "val_steps": 10,
        "weight_path": "weights/model_{epoch:05d}.h5",
        "alpha": 0.01,  # loss mulltiplier
        "vgg_layer_nums": [2, 5, 9],
        "tiled": False,  # True of False
        "model": "fsrcnn",  # "fsrcnn" or "fsrcnn_tiled"
        "pretrainedWeightPath": ""
    }

    train(training_config)
