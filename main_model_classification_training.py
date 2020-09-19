import path_configs # noqa
import settings
import os
import time
import tensorflow as tf
settings.init()
from EncoderGenerators import (TrainEncoderGenerator, # noqa
                               ValidationEncoderGenerator)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def scaled(tensor):
    return (tensor-tf.math.reduce_min(tensor))/(tf.math.reduce_max(tensor)-tf.
                                                math.reduce_min(tensor))


def doubleOutput(input):
    return input, input