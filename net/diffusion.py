from util import utils
from tensorflow.python.keras import Model
import tensorflow as tf

class GaussianDiffusion(Model):
    '''
    高斯扩散模型
    '''

    def __init__(self,
                 betas,
                 ):
        super(GaussianDiffusion, self).__init__()
        self.betas = betas

    def add_nosie_times(self, input_shape):
        gaussian_noise = utils.add_gaussian_noise(input_shape)
        times = utils.add_randint_noise(input_shape[0])
        return gaussian_noise, times

    def perturb_x(self, x, t, noise):
        x = tf.cast(x,dtype=tf.float32)

        _, _, alphas_cumprod = utils.alphas_cumprod(self.betas)

        signa_rate, noise_rate = utils.signa_rate(alphas_cumprod), utils.nosie_rate(alphas_cumprod)

        out = utils.extract(signa_rate, t, x.shape) * x + utils.extract(noise_rate, t, x.shape) * noise  # 正向扩散的公式

        return out

    def call(self, inputs, training=None, mask=None):
        gaussian_noise, times = self.add_nosie_times(inputs.shape)  # to get gaussian and times

        perturbed_x = self.perturb_x(inputs, times, gaussian_noise)  # 扩散模型加入噪声

        return perturbed_x, gaussian_noise, times

# if __name__ == '__main__':
    # x = tf.random.normal(shape=(128, 32, 32, 3))
    #
    # betas = utils.generate_cosine_schedule(cig.T)
    # # plt.plot(np.arange(len(betas)),betas)
    # # plt.show()
    #
    # model = GaussianDiffusion(betas=betas,
    #                           times_encoder_channels=cig.TIMES_CHANNELS,
    #                           iniconv_channels=cig.INICONV_CHANNELS,
    #                           unet_channels=(cig.UNET_CHANNELS_DOWN, cig.UNET_CHANNELS_MID, cig.UNET_CHANNELS_UP),
    #                           dropout_rate=cig.DROPOUT_RATE,
    #                           use_attention=(cig.USE_ATTENTION_DOWN, cig.USE_ATTENTION_MID, cig.USE_ATTENTION_UP),
    #                           self_attention=(cig.SELF_ATTENTION_DOWN, cig.SELF_ATTENTION_MID, cig.SELF_ATTENTION_UP),
    #                           num_batch=cig.BN_NUM_BATCH,
    #                           activate=cig.ACTIVATE[0])
    #
    # predit_noise, gaussian_noise = model(x,)
    # model.summary()
    # print(predit_noise.shape, gaussian_noise.shape)
