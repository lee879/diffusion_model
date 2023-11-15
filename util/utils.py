import cig
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

@tf.function
def generate_cosine_schedule(T, s=0.008):
    '''
    :param T: 信号率
    :param s: 放缩系数
    :return: 返回的是[a0,a1,a2,a3,a4,a5.....]
    '''

    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return tf.cast(tf.convert_to_tensor(betas),dtype=tf.float32)


def generate_linear_schedule(T):
    '''
    :param T: 信号率
    :return: 返回的是[a0,a1,a2,a3,a4,a5.....]
    '''
    low = cig.SCHEDELE_LOW * 1000 / T,
    high = cig.SCHEDELE_HIGH * 1000 / T,
    out = np.linspace(low, high, T)
    return tf.cast(tf.convert_to_tensor(out),dtype=tf.float32)

# def extract(a, t, x_shape):
#     '''
#     :param a: 生成的每一个时刻的alphas
#     :param t: 获得每一个时刻
#     :param x_shape:
#     :return: 对应时刻的alphas
#     '''
#     a_copy = np.copy(a)
#     t_copy = np.copy(t)
#
#     b, *_ = t_copy.shape
#
#     # 使用 np.take 获取对应时刻的alphas
#     out = np.take(a_copy, t_copy, axis=-1)
#
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def extract(a, t, x_shape):
    '''
    :param a: 生成的每一个时刻的 alphas
    :param t: 获得每一个时刻
    :param x_shape:
    :return: 对应时刻的 alphas
    '''
    # 创建 TensorFlow 副本
    a_copy = tf.identity(a)
    t_copy = tf.identity(t)

    b, *_ = t_copy.shape

    # 使用 tf.gather 获取对应时刻的 alphas
    out = tf.gather(a_copy, t_copy, axis=-1)

    # 重塑张量形状
    out = tf.reshape(out, (b,) + (1,) * (len(x_shape) - 1))

    return out

def add_gaussian_noise(images_shape):
    '''
    :param images_shape: images_shape
    :return: noise
    '''
    return tf.random.normal(shape=images_shape)

def add_randint_noise(batch):
    '''
    :return: times
    '''
    return tf.convert_to_tensor(np.random.randint(low=0, high=cig.T, size=(batch,)))

def add_simple_int_noise(batch):
    random_value = tf.random.uniform([1], 0, cig.T, dtype=tf.int32)
    out = tf.fill([batch], random_value[0])
    return out

def alphas_cumprod(betas):
    '''
    :param betas:
    :return: betas, alphas, alphas_cumprod
    '''
    alphas = 1 - betas
    alphas_cd = tf.math.cumprod(alphas)  # a0a1a2a3a4a5
    return betas, alphas, alphas_cd

def remove_noise_coeff(betas, cumprod_alphas):
    out = betas / tf.sqrt(1 - cumprod_alphas)
    return out

def signa_rate(alphas_cumprod):
    return tf.math.sqrt(alphas_cumprod)

def nosie_rate(alphas_cumprod):
    return tf.math.sqrt(1 - alphas_cumprod)

def remove_noise_coeff(betas):
    _,_,alphas_cd = alphas_cumprod(betas)
    return betas / tf.sqrt(1 - alphas_cd)

def reciprocal_sqrt_alphas(betas):
    _, alphas, _ = alphas_cumprod(betas)
    return 1 / tf.sqrt(alphas)

# 时间打印函数
def time_print(use_time):
    use_time_in_seconds = int(use_time)
    hours = use_time_in_seconds // 3600
    minutes = (use_time_in_seconds % 3600) // 60
    seconds = use_time_in_seconds % 60
    print(f"本次训练经过的时间为: {hours} 小时, {minutes} 分钟, {seconds} 秒")

# ddim 采样方式
def ddim_times_set():

    alpha_times = tf.cast(tf.range(0, cig.T, cig.T // cig.DDIM_SAMPLE_TIMES),dtype=tf.int32)

    alpha_times_pre = tf.concat([tf.constant([0], dtype=tf.int32), alpha_times[:-1]], axis=0)

    return alpha_times, alpha_times_pre

#预测的参数
def ddim_predict_para_0(betas):

    _, _, alphas_cd = alphas_cumprod(betas)

    return tf.sqrt(alphas_cd)

#预测的参数
def ddim_predict_para_1(betas):
    _, _, alphas_cd = alphas_cumprod(betas)
    return tf.sqrt(1 - alphas_cd)


def ddim_sigma(x,betas,times,times_pre):

    para_0 = extract(ddim_predict_para_1(betas),times_pre,x_shape=x.shape) / extract(ddim_predict_para_1(betas),times,x.shape)
    para_1 = 1 - extract(ddim_predict_para_0(betas),times,x.shape) / extract(ddim_predict_para_0(betas),times_pre,x.shape)

    return tf.sqrt(para_0 * para_1) * cig.DDIM_ETA

# 预测方向系数
def ddim_predict_direc(x,betas,times,times_pre):
   # t = cig.DDIM_ETA * ddim_sigma(x,betas,times,times_pre)
    _, _, alphas_cd = alphas_cumprod(betas)

    return tf.sqrt(extract(1 - alphas_cd,times_pre,x.shape) - ddim_sigma(x,betas,times,times_pre))
#ddim_times_set()