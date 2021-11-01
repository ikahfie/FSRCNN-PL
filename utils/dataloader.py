import os
import albumentations as A
import numpy as np
import random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import cast, float16, float32
from tensorflow.keras.utils import Sequence
try:
    from utils.constants import *
except:
    from constants import *
from tensorflow.compat.v1.logging import set_verbosity, ERROR
set_verbosity(ERROR)


class DIV2K(Sequence):
    def __init__(self,
                 hr_image_folder: str,
                 batch_size: int,
                 set_type: str) -> None:

        self.batch_size = batch_size
        self.hr_image_folder = hr_image_folder
        self.image_fns = np.sort([
            x for x in os.listdir(self.hr_image_folder) if x.endswith(IMAGE_FORMAT)
        ])

        if set_type == "TRAIN":
            self.image_fns = self.image_fns[:200]
        elif set_type == "VAL":
            self.image_fns = self.image_fns[-200:-100]
        elif set_type == "ALL":
            self.image_fns = np.concatenate(
                (self.image_fns[:200],
                self.image_fns[-200:100]),
                axis=0)
        else:
            self.image_fns = self.image_fns[-100]

        if set_type in ["train", "val"]:
            self.transform = A.Compose(
                [
                    A.RandomCrop(
                        width=HR_IMG_SIZE[0], 
                        height=HR_IMG_SIZE[1], 
                        p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(
                        brightness=0.1, 
                        contrast=0.1, 
                        saturation=0.1, 
                        hue=0.1, 
                        p=0.8
                    ),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.RandomCrop(
                        width=HR_IMG_SIZE[0], 
                        height=HR_IMG_SIZE[1], 
                        p=1.0),
                ]
            )

        self.to_float = A.ToFloat(max_value=255)

    def __len__(self):
        return len(self.image_fns) // self.batch_size

    def on_epoch_end(self):
        random.shuffle(self.image_fns)

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_image_fns = self.image_fns[i:i + self.batch_size]
        batch_HR_images = np.zeros(shape=(
            self.batch_size, 
            HR_IMG_SIZE[0], 
            HR_IMG_SIZE[1], 
            COLOR_CHANNELS))
        batch_LR_images = np.zeros(shape=(
            self.batch_size, 
            LR_IMG_SIZE[0], 
            LR_IMG_SIZE[1], 
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

            # batch_HR_images[i] = self.to_float(image=hr_image_transform)["image"]
            # batch_LR_images[i] = self.to_float(image=lr_image_transform)["image"]

            batch_HR_images[i] = cast(
                hr_image_transform/255.,
                dtype=float32)
            batch_LR_images[i] = cast(
                lr_image_transform/255.,
                dtype=float32)

            return (batch_LR_images, batch_HR_images)
