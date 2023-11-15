import tensorflow as tf
import numpy as np


# l2
def mse_fd(perdict,label):
    return tf.reduce_mean(tf.square(perdict - label))
# l1
def l1_fd(perdict,label):
    return tf.reduce_mean(tf.abs(perdict,label))


class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, T):
        super(CosineAnnealingSchedule, self).__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T = T

    def __call__(self, step):
        t = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos((step/self.T) * np.pi))
        return t