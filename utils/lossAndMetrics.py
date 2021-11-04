import os

os.environ["TF_CPP_LOG_LEVEL"] = "3"
from tensorflow import function as tf_f
from tensorflow.python.keras.losses import MeanAbsoluteError
from tensorflow.python.ops.math_ops import (
    reduce_mean,
    reduce_sum,
    subtract,
    abs
)
from tensorflow.python.ops.image_ops import ssim, psnr
from tensorflow.keras.applications.vgg16 import preprocess_input

from utils.model import vgg_loss

def SSIM(y_true, y_pred):
    """
    computes SSIM between y_true and y_pred.

    Args:
    :y_true true label class
    :y_pred predicted label class
    :return SSIM Value

    """

    return ssim(y_true, y_pred, max_val=1.0)

def PSNR(y_true, y_pred):
    """
    computes SSIM between y_true and y_pred.

    Args:
    :y_true true label class
    :y_pred predicted label class
    :return SSIM Value

    """

    return psnr(y_true, y_pred, max_val=1.0)


def __distance(y_true, y_pred, loss_model):
    """
    Calculate absolute distance between y_true,y_pred.

    :param y_true: true label class
    :param y_pred: predicted label class
    :param loss_model: VGG loss network
    :return: mean of abs distance between loss network features.
    """
    input0 = loss_model(y_true)
    input1 = loss_model(y_pred)

    l1 = abs(subtract(input0, input1))
    l1 = reduce_sum(l1, axis=-1)

    return reduce_mean(l1, axis=(1, 2))


def perceptual_loss(alpha: float, vgg_layer_nums: list, tiled: bool):
    """
    Function wrapper for the perceptual loss function.
    declaring loss_models and mae.

    :param alpha: weight of VGG losses
    :param vgg_layer_nums: list of indexes of choosen VGG Layer
    :param tiled: if input is tiled or not?
    :return: loss function
    """
    loss_models = vgg_loss(vgg_layer_nums, tiled)
    mae = MeanAbsoluteError()

    @tf_f
    def loss(y_true, y_pred):
        """
        Calculate perceptual loss based on batch y_true and y_pred.
        - Preproc-ing both y_true and y_pred with tf.keras.applications.vgg16.preprocess_input()
        - Calculating both
        :param y_true:
        :param y_pred:
        :return:
        """
        y_true = preprocess_input(y_true)
        y_pred = preprocess_input(y_pred)

        __loss = []
        for model in loss_models:
            __loss.append(
                __distance(
                    y_true,
                    y_pred,
                    model
                )
            )

        return alpha * ((reduce_sum(__loss) + reduce_sum(mae(y_true, y_pred))))

    return loss
