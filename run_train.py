try:
    from utils.train import train
except ModuleNotFoundError:
    from train import train

config = {
        "dataset_dir": "dataset\DIV2K_train_HR",
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
        "alpha_loss": 0.01
    }
if __name__=="__main__":
    train(config)