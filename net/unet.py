import tensorflow as tf
from styleganplus.self_layers import UpsampleBlock,burpool
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D
from .self_layers import ResidualBlock, Dowm, Up, SelectActivate, TimesEncoder, IniConv

class Unet(Model):
    def __init__(self,
                 channels,
                 times_encoder_channels,
                 iniconv_channels,
                 dropout_rate,
                 use_attention,
                 self_attention,
                 num_batch,
                 activate,
                 times_emb_channels
                 ):
        super(Unet, self).__init__()
        self.channels = channels
        self.times_encoder = TimesEncoder(out_channels=times_encoder_channels,times_emb_channels=times_emb_channels)
        self.iniconv = IniConv(out_channels=iniconv_channels)
        self.residualblock_0 = [ResidualBlock(channels=channel,
                                              dropout_rate=dropout_rate,
                                              use_attention=attention,
                                              self_attention=attention_self,
                                              num_groups=num_batch,
                                              activate=activate)
                                for channel, attention, attention_self in
                                zip(self.channels[0], use_attention[0], self_attention[0])]

        self.residualblock_1 = [ResidualBlock(channels=channel,
                                              dropout_rate=dropout_rate,
                                              use_attention=attention,
                                              self_attention=attention_self,
                                              num_groups=num_batch,
                                              activate=activate)
                                for channel, attention, attention_self in
                                zip(self.channels[1], use_attention[1], self_attention[1])]

        self.residualblock_2 = [ResidualBlock(channels=channel,
                                              dropout_rate=dropout_rate,
                                              use_attention=attention,
                                              self_attention=attention_self,
                                              num_groups=num_batch,
                                              activate=activate)
                                for channel, attention, attention_self in
                                zip(self.channels[2], use_attention[2], self_attention[2])]

        self.down = [Dowm(self.channels[0][i]) for i in [0, 2, 4, 6]]  # 标准的unet下采样模块
        #self.down = [burpool(self.channels[0][i]) for i in [0, 2, 4, 6]]  # 标准的unet下采样模块
        self.up = [Up(self.channels[2][i]) for i in [0, 2, 4, 6]]  # 标准的unet上采样模块
        #self.up = [UpsampleBlock(self.channels[2][i]) for i in [0, 2, 4, 6]]  # 标准的unet上采样模块

        # 得到预测噪声模块
        #self.bn = BatchNormalization()
        self.ac = SelectActivate(activate=activate)
        self.out_conv = Conv2D(3,1,1,"same",dtype=tf.float32)

    def call(self, inputs, times, training=None,mask=None):
        time_encoder = self.times_encoder(times)  # 进行时间的编码
        image_encoder = self.iniconv(inputs)

        ferture_map_list = []

        # down
        temp_i = 0
        for i in range(len(self.channels[0]) // 2):
            l0_0 = self.residualblock_0[temp_i](image_encoder, time_encoder)
            l0_1 = self.residualblock_0[temp_i + 1](l0_0, time_encoder)
            ferture_map_list.append(l0_1)
            image_encoder = self.down[i](l0_1)
            temp_i += 2

        # mid
        temp_i = 0
        l1_0 = self.residualblock_1[temp_i](image_encoder, time_encoder)
        image_encoder = self.residualblock_1[temp_i + 1](l1_0, time_encoder)

        # up
        temp_i = 0
        for i in range(len(self.channels[2]) // 2):
            l2_0 = self.up[i](image_encoder)
            l2_1 = self.residualblock_2[temp_i](tf.concat((ferture_map_list[3 - i], l2_0), axis=-1), time_encoder)
            image_encoder = self.residualblock_2[temp_i + 1](l2_1, time_encoder)
            temp_i += 2
        out = self.out_conv(self.ac(image_encoder))
        return out

# if __name__ == '__main__':

    # x = tf.random.normal(shape=(8, 128, 128, 64))
    # t = tf.random.normal(shape=(8,))
    # model = Unet(channels=(cig.UNET_CHANNELS_DOWN, cig.UNET_CHANNELS_MID, cig.UNET_CHANNELS_UP),
    #              dropout_rate=cig.DROPOUT_RATE,
    #              use_attention=(cig.USE_ATTENTION_DOWN, cig.USE_ATTENTION_MID, cig.USE_ATTENTION_UP),
    #              self_attention=(cig.SELF_ATTENTION_DOWN, cig.SELF_ATTENTION_MID, cig.SELF_ATTENTION_UP),
    #              num_batch=cig.BN_NUM_BATCH,
    #              activate=cig.ACTIVATE[0])
    #
    # y = model(x, t)
    # print(y.shape)
