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

try:
    from utils.model import vgg_loss
except ModuleNotFoundError:
    from model import vgg_loss


def metric_ssim(y_true, y_pred):
    """
    computes SSIM between y_true and y_pred.

    Args:
    :y_true true label class
    :y_pred predicted label class
    :return: SSIM Value

    """

    return ssim(y_true, y_pred, max_val=1.0)


def metric_psnr(y_true, y_pred):
    """
    computes SSIM between y_true and y_pred.

    Args:
    :y_true true label class
    :y_pred predicted label class
    :return SSIM Value

    """

    return psnr(y_true, y_pred, max_val=1.0)


def __distance(y_true, y_pred, loss_model):
    input0 = loss_model(y_true)
    input1 = loss_model(y_pred)

    l1 = abs(subtract(input0, input1))
    l1 = reduce_sum(l1, axis=-1)

    return reduce_mean(l1, axis=(1, 2))


def perceptual_loss(alpha: float, vgg_layer_nums: list, tiled: bool):
    loss_models = vgg_loss(vgg_layer_nums, tiled)
    mae = MeanAbsoluteError()

    @tf_f
    def loss(y_true, y_pred):
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

        return alpha * (reduce_sum(__loss) + reduce_mean(mae(y_true, y_pred)))

    return loss
