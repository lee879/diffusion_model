import tensorflow as tf
from tensorflow.python.keras import Model
from .diffusion import GaussianDiffusion
from util import utils
from util.utils import ddim_times_set
from tqdm import tqdm
from .unet import Unet

class DDIM(Model):
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
                 **args
                 ):
        super(DDIM, self).__init__()
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
                               activate=activate,
                               times_emb_channels=args["time_emb_channels"]
                               )

    def remove_noise(self, x, times,times_pre):

        perdict_0 = utils.extract(utils.ddim_predict_para_0(self.betas), times_pre, x.shape)
        perdict_1 = utils.extract(utils.ddim_predict_para_1(self.betas), times, x.shape)
        perdict_2 = utils.extract(utils.ddim_predict_para_0(self.betas), times, x.shape)

        perdict = perdict_0 * (x - perdict_1 * self.unet_model(x,times)) / perdict_2
        dire = utils.ddim_predict_direc(x,self.betas,times,times_pre) * self.unet_model(x,times)

        out = perdict + dire

        return out


    def sample(self, num_sample):
        assert isinstance(num_sample, int), \
            "input is not int"

        x = utils.add_gaussian_noise(images_shape=(num_sample, self.image_size, self.image_size, 3))  # 生成一个batch的输入噪声
        alpha_times, alpha_times_pre = ddim_times_set()
        for t in tqdm(reversed(range(0, len(alpha_times))), desc='sampling loop time step', total=len(alpha_times)):

            times = tf.fill([num_sample], alpha_times[t])  # 得到次数
            times_pre = tf.fill([num_sample], alpha_times_pre[t])

            x = self.remove_noise(x, times,times_pre)

            if t >= 0:
                x += utils.ddim_sigma(x, self.betas, times, times_pre) * utils.add_gaussian_noise(x.shape)
        return x

    def call(self, train_inputs, sample, test=False, training=None, mask=None):
        if test:
            out = self.sample(num_sample=sample)
            return out

        perturbed_x, gaussian_noise, times = self.diffusion(train_inputs)

        predict_noise = self.unet_model(perturbed_x, times)

        return predict_noise, gaussian_noise

# if __name__ == '__main__':
#     import cig
#     x = tf.random.normal(shape=(128, 32, 32, 3))
#
#     betas = utils.generate_cosine_schedule(cig.T)
#     # plt.plot(np.arange(len(betas)),betas)
#     # plt.show()
#
#     model = DDIM(betas=betas,
#                  image_size=32,
#                  times_encoder_channels=cig.TIMES_CHANNELS,
#                  iniconv_channels=cig.INICONV_CHANNELS,
#                  channels=(cig.UNET_CHANNELS_DOWN, cig.UNET_CHANNELS_MID, cig.UNET_CHANNELS_UP),
#                  dropout_rate=cig.DROPOUT_RATE,
#                  use_attention=(cig.USE_ATTENTION_DOWN, cig.USE_ATTENTION_MID, cig.USE_ATTENTION_UP),
#                  self_attention=(cig.SELF_ATTENTION_DOWN, cig.SELF_ATTENTION_MID, cig.SELF_ATTENTION_UP),
#                  num_batch=cig.BN_NUM_BATCH,
#                  activate=cig.ACTIVATE[0],
#                  T=cig.T)
# tt,yy = model(x, 10, True)
#
# model.summary()
# print(tt.shape,yy.shape)
