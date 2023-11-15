import tensorflow as tf
from tensorflow.python.keras import Model
from .diffusion import GaussianDiffusion
from util import utils
from tqdm import tqdm
from .unet import Unet

class DDPM(Model):
    '''
    to use simple model ,get good performerce
    '''

    def __init__(self,
                 betas,
                 image_size,
                 channels,
                 times_encoder_channels,
                 iniconv_channels,
                 dropout_rate,
                 use_attention,
                 self_attention,
                 num_batch,
                 activate,
                 T,
                 ):
        super(DDPM, self).__init__()
        self.T = T
        self.betas = betas
        self.image_size = image_size
        self.diffusion = GaussianDiffusion(betas=betas)
        self.unet_model = Unet(channels=channels,
                               times_encoder_channels=times_encoder_channels,
                               iniconv_channels=iniconv_channels,
                               dropout_rate=dropout_rate,
                               use_attention=use_attention,
                               self_attention=self_attention,
                               num_batch=num_batch,
                               activate=activate)

    def remove_noise(self, x, times):

        return (
                utils.extract(utils.reciprocal_sqrt_alphas(self.betas), times, x.shape) *
                (x - utils.extract(utils.remove_noise_coeff(self.betas), times, x.shape) * self.unet_model(x,times))
            )

    def sample(self, num_sample):
        assert isinstance(num_sample, int), \
            "input is not int"

        x = utils.add_gaussian_noise(images_shape=(num_sample, self.image_size, self.image_size, 3))  # 生成一个batch的输入噪声

        for t in tqdm(range(self.T - 1, -1, -1),position=0,leave=True):
            times = tf.fill([num_sample], t)  # 得到次数
            x = self.remove_noise(x, times)
            if t >= 0:
                x += utils.extract(utils.sigma(self.betas), times, x.shape) * utils.add_gaussian_noise(x.shape)
        return x

    def call(self, train_inputs, sample, test=False, training=None, mask=None):
        if test:
            out = self.sample(num_sample=sample)
            return out

        perturbed_x, gaussian_noise, times = self.diffusion(train_inputs)

        predict_noise = self.unet_model(perturbed_x, times)

        return predict_noise, gaussian_noise

# if __name__ == '__main__':
#     x = tf.random.normal(shape=(128, 32, 32, 3))
#
#     betas = utils.generate_cosine_schedule(cig.T)
#     # plt.plot(np.arange(len(betas)),betas)
#     # plt.show()
#
#     model = DDPM(betas=betas,
#                  image_size=32,
#                  times_encoder_channels=cig.TIMES_CHANNELS,
#                  iniconv_channels=cig.INICONV_CHANNELS,
#                  unet_channels=(cig.UNET_CHANNELS_DOWN, cig.UNET_CHANNELS_MID, cig.UNET_CHANNELS_UP),
#                  dropout_rate=cig.DROPOUT_RATE,
#                  use_attention=(cig.USE_ATTENTION_DOWN, cig.USE_ATTENTION_MID, cig.USE_ATTENTION_UP),
#                  self_attention=(cig.SELF_ATTENTION_DOWN, cig.SELF_ATTENTION_MID, cig.SELF_ATTENTION_UP),
#                  num_batch=cig.BN_NUM_BATCH,
#                  activate=cig.ACTIVATE[0],
#                  T=cig.T)
# tt,yy = model(x, 10, False)
#
# model.summary()
# print(tt.shape,yy.shape)
