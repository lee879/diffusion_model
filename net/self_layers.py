import tensorflow as tf
from tensorflow.python.keras.layers import Dense,  Conv2D, Layer, \
    Conv2DTranspose, Activation, Dropout, Lambda

class Dowm(Layer):
    '''
    2倍下采样
    '''

    def __init__(self, filts):
        super(Dowm, self).__init__()
        self.conv = Conv2D(filts, 1, 2, "same")

    def call(self, inputs, **kwargs):
        out = self.conv(inputs)
        return out


class Up(Layer):
    '''
    2倍上采样
    '''

    def __init__(self, filts):
        super(Up, self).__init__()
        self.ConvTran = Conv2DTranspose(filts, 3, 2, "same")

    def call(self, inputs, **kwargs):
        out = self.ConvTran(inputs)
        return out


class IniConv(Layer):
    '''
    卷积初始化层
    '''

    def __init__(self, out_channels):
        super(IniConv, self).__init__()

        self.out_channels = out_channels
        self.iniconv = Conv2D(self.out_channels, 3, 1, "same")

    def call(self, inputs, **kwargs):
        return self.iniconv(inputs)


class GroupNormalization(Layer):
    def __init__(self,
                 num_batch: int,
                 eps: float = 1e-6,
                 ):
        super(GroupNormalization, self).__init__()
        self.num_batch = num_batch
        self.eps = eps

    def build(self, input_shape):
        assert input_shape[-1] % self.num_batch == 0, \
            "you num_groups set eorror"
        self.num_groups = input_shape[-1] // self.num_batch

        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],),
                                     initializer=tf.keras.initializers.Constant(value=1.0),
                                     trainable=True)
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],),
                                    initializer=tf.keras.initializers.Constant(value=0.0),
                                    trainable=True)
        super(GroupNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        group_size = input_shape[-1] // self.num_groups
        grouped_inputs = tf.reshape(inputs, [-1, self.num_groups, group_size])

        group_mean = tf.reduce_mean(grouped_inputs, axis=[-1], keepdims=True)
        group_var = tf.reduce_mean(tf.square(grouped_inputs - group_mean), axis=[-1], keepdims=True)

        normalized_inputs = (grouped_inputs - group_mean) / tf.sqrt(group_var + self.eps)

        normalized_inputs = tf.reshape(normalized_inputs, input_shape)

        normalized_output = self.gamma * normalized_inputs + self.beta

        return normalized_output


class TimesEncoder(Layer):
    '''
    次数编码层
    '''

    def __init__(self, out_channels,times_emb_channels, scal=1.0):
        super(TimesEncoder, self).__init__()
        self.out_channels = out_channels
        self.scal = scal
        self.l0 = Dense(self.out_channels)
        self.ac = Activation(tf.nn.silu)
        self.l1 = Dense(self.out_channels)
        self.times_emb_channels = times_emb_channels

    def endering(self, x):
        # Apply tf.stop_gradient to prevent gradient computation for the input
        half_dim = self.times_emb_channels
        emb = tf.math.log(tf.constant(10000, dtype=tf.float32)) / tf.cast(half_dim, dtype=tf.float32)
        emb = tf.exp(tf.cast(tf.range(half_dim), dtype=tf.float32) * -emb)
        emb = tf.tensordot(tf.cast(x, dtype=tf.float32) * tf.cast(self.scal, dtype=tf.float32), emb, axes=0)
        emb = tf.concat((tf.sin(emb), tf.cos(emb)), axis=-1)
        return emb

    def call(self, times, **kwargs):
        emb_times = self.endering(times)
        out = self.l0(emb_times)
        out = self.ac(out)
        out = self.l1(out)
        return out

class VitBlock(Layer):
    def __init__(self):
        super(VitBlock, self).__init__()

    def call(self, inputs, **kwargs):
        return None


class AttentionBlock(Layer):
    '''
    注意力层：包括使用自注意力层 , 计算（q,k,v）矩阵的成本很高
    '''

    def __init__(self, in_channels, num_groups, self_attention):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.norm = GroupNormalization(num_batch=num_groups)
        self.to_qkv = tf.keras.layers.Conv2D(in_channels * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = tf.keras.layers.Conv2D(in_channels, kernel_size=1, strides=1, use_bias=False)
        self.self_attention = self_attention

    def call(self, x):
        b, h, w, c = x.shape
        q, k, v = tf.split(self.to_qkv(self.norm(x)), num_or_size_splits=3, axis=-1)
        if self.self_attention:
            temp = tf.add(tf.add(v, q), k) / 3
            q = k = v = temp

        q = tf.transpose(q, perm=[0, 3, 1, 2])
        q = tf.reshape(q, (b, h * w, c))
        k = tf.reshape(k, (b, c, h * w))
        v = tf.transpose(v, perm=[0, 3, 1, 2])
        v = tf.reshape(v, (b, h * w, c))

        dot_products = tf.matmul(q, k, transpose_b=False) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = tf.nn.softmax(dot_products, axis=-1)
        out = tf.matmul(attention, v, transpose_a=float)
        assert out.shape == (b, h * w, c)
        out = tf.reshape(out, (b, h, w, c))

        return self.to_out(out) + x

class ResidualBlock(Layer):
    '''
    自定义残差层
    '''

    def __init__(self,
                 channels,
                 dropout_rate,
                 use_attention,
                 self_attention,
                 num_groups,
                 activate,
                 ):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.use_attention = use_attention
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv_1 = Conv2D(channels, 3, 1, "same")
        self.ac = SelectActivate(activate=activate)
        self.times_bias = Dense(channels)
        self.dp = Dropout(dropout_rate)
        self.conv_2 = Conv2D(channels, 3, 1, "same")
        self.attention = AttentionBlock(in_channels=channels, num_groups=num_groups, self_attention=self_attention)

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            self.residual_connection = Conv2D(self.channels, 1, 1, "same")
        else:
            self.residual_connection = Lambda(lambda x: x)  # 使用Lambda层来创建恒等映射

    def call(self, x, t, **kwargs):
        x0 = self.ac(self.bn1(x))
        x1 = self.conv_1(x0)
        x2 = x1 + self.ac(self.times_bias(t))[:, None, None, :]  # 位置编码和数的融合
        x3 = self.ac(self.bn2(x2))
        x4 = self.conv_2(self.dp(x3)) + self.residual_connection(x)
        if self.use_attention:
            out = self.attention(x4)
            return out
        return x4

class SelectActivate(Layer):
    def __init__(self,activate):
        super(SelectActivate, self).__init__()
        if activate == "relu":
            self.ac = Activation(tf.nn.relu)
        elif activate == "swich":
            self.ac = Activation(tf.nn.swish)
        elif activate == "tanh":
            self.ac = Activation(tf.nn.tanh)
        else:
            self.ac = Activation(tf.nn.relu)

    def call(self, inputs, **kwargs):

        return self.ac(inputs)

# x = tf.random.normal(shape=(128,16,16,64))
# t = tf.random.normal(shape=(128,512))
#
# model = ResidualBlock(channels=64,
#                  dropout_rate=0.2,
#                  use_attention=True,
#                  self_attention=False,
#                  num_groups=4)
# y = model(x,t)
# print(y.shape)
