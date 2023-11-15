import tensorflow as tf
# class EMA:
#     def __init__(self, decay):
#         self.decay = decay
#     def update_average(self, old, new):
#         if old is None:
#             return new
#         return old * self.decay + (1 - self.decay) * new
#
#     def update_model_average(self, ema_model, current_model):
#
#         for i,(ema_params, current_params) in enumerate(zip(ema_model.trainable_variables, current_model.trainable_variables)):
#             old, new = ema_params,current_params
#             ema_params.assign(self.update_average(old, new))

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.ema = tf.train.ExponentialMovingAverage(decay=decay)

    def update_model_average(self, ema_model, current_model):
        # 在当前模型的所有可训练变量上建立指数移动平均
        self.ema.apply(current_model.trainable_variables)

        # 将指数移动平均的值赋给EMA模型的变量
        for ema_params, current_params in zip(ema_model.trainable_variables, current_model.trainable_variables):
            ema_params.assign(self.ema.average(current_params))