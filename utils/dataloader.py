import os

import albumentations as al
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import cast, float32
from tensorflow.keras.utils import Sequence
from tensorflow.compat.v1.logging import set_verbosity, ERROR


import sys
sys.path.append(r"../")
from image_config import *

set_verbosity(ERROR)


class DIV2K(Sequence):
    def __init__(self,
                 hr_image_folder: str,
                 batch_size: int,
                 set_type: str,
                 tiled: bool) -> None:

        self.tiled = tiled
        if self.tiled:
            self.LR_IMG_SIZE = LR_TILE_SIZE
            self.HR_IMG_SIZE = HR_TILE_SIZE

        self.batch_size = batch_size
        self.hr_image_folder = hr_image_folder
        self.image_fns = np.sort([
            x for x in os.listdir(self.hr_image_folder) if x.endswith(IMAGE_FORMAT)
        ])

        if set_type in ["train", "val"]:
            self.transform = al.Compose(
                [al.RandomCrop(
                    width=self.HR_IMG_SIZE[0],
                    height=self.HR_IMG_SIZE[1],
                    p=1.0),
                    al.HorizontalFlip(p=0.5),
                    al.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.1,
                        p=0.8
                    )
                ]
            )
        else:
            self.transform = al.Compose(
                [
                    al.RandomCrop(
                        width=self.HR_IMG_SIZE[0],
                        height=self.HR_IMG_SIZE[1],
                        p=1.0),
                ]
            )

        self.to_float = al.ToFloat(max_value=255)

    def __len__(self):
        return len(self.image_fns) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.image_fns)

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_image_fns = self.image_fns[i:i + self.batch_size]
        batch_hr_images = np.zeros(shape=(
            self.batch_size,
            self.HR_IMG_SIZE[0],
            self.HR_IMG_SIZE[1],
            COLOR_CHANNELS))
        batch_lr_images = np.zeros(shape=(
            self.batch_size,
            self.LR_IMG_SIZE[0],
            self.LR_IMG_SIZE[1],
            COLOR_CHANNELS))

        for i, image_fn in enumerate(batch_image_fns):
            hr_image_pil = Image.open(
                os.path.join(
                    self.hr_image_folder, image_fn))

            hr_image = np.array(hr_image_pil)

            hr_image_transform = self.transform(image=hr_image)["image"]
            hr_image_transform_pil = Image.fromarray(hr_image_transform)
            lr_image_transform_pil = hr_image_transform_pil.resize(
                LR_IMG_SIZE, resample=DOWNSAMPLE_MODE
            )
            lr_image_transform = np.array(lr_image_transform_pil)

            # batch_hr_images[i] = self.to_float(image=hr_image_transform)["image"]
            # batch_lr_images[i] = self.to_float(image=lr_image_transform)["image"]

            batch_hr_images[i] = cast(
                hr_image_transform / 255.,
                dtype=float32)
            batch_lr_images[i] = cast(
                lr_image_transform / 255.,
                dtype=float32)

        return batch_lr_images, batch_hr_images
