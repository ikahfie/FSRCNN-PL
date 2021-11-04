from image_config import HR_TILE_SIZE


try:
    from utils.train import train
except ModuleNotFoundError:
    from train import train

training_config = {
    "train_dir": "dataset/DIV2K_train_HR",
    "val_dir": "dataset/DIV2K_valid_HR",
    "d": 32,
    "s": 5,
    "m": 1,
    "lr": 0.001,  # learning rate
    "epoch": 500,
    "batch_size": 32,
    "steps_per_epoch": 4,
    "val_batch_size": 32,
    "val_steps": 10,
    "weight_path": "weights/model_{epoch:05d}.h5",
    "alpha": 1.0,  # loss mulltiplier
    "vgg_layer_nums": [2, 5, 9],
    "tiled": False,
    "model": "fsrcnn",  # "fsrcnn" or "fsrcnn_tiled"
    "pretrainedWeightPath": ""
}

