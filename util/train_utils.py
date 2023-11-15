import os
import random
import string
from util import utils
import tensorflow as tf
from loss.loss_fd import mse_fd,l1_fd

#混合精度训练代码
# 如果你的GPU不支持混合精度训练，这边建议不要开启哟
def mixed_pre(select):
    '''
    :param select:
    :return:
    '''
    if select:  # 开启混合精度训练
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
        poliy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(poliy)
        print("---------------混合精度训练开启---------------")
    else:
        # 关闭混合精度训练
        # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
        # poliy = tf.keras.mixed_precision.Policy("float32")
        # tf.keras.mixed_precision.set_global_policy(poliy)poliy
        print("---------------不使用混合精度训练---------------")

# 混合精度优化器选择
def opt(select, lr, eps, beta1, beta2):
    if select:  # 混合精度训练器
        return tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps, beta_1=beta1, beta_2=beta2))
    else:  # 普通的优化器
        return tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps, beta_1=beta1, beta_2=beta2)

# 得到betas
def get_betas(schedele,t):
    if schedele == "linear":
        betas = utils.generate_linear_schedule(t)
    else:
        betas = utils.generate_cosine_schedule(t)
    return betas

# 进行梯度跟踪
def gd_record(train_image, model, sample, test,select_loss, regularization, regularization_strength, elastic_eta):
    with tf.GradientTape() as tape:
        predict_noise, label_noise = model(train_image, sample, test)
        if select_loss == "L1":
            loss = l1_fd(predict_noise, label_noise)
            if regularization is not None:
                reg_loss = reg(model, regularization=regularization, regularization_strength=regularization_strength, elastic_eta=elastic_eta)
                loss += reg_loss
        else:
            loss = mse_fd(predict_noise, label_noise)
            if regularization is not None:
                reg_loss = reg(model, regularization=regularization, regularization_strength=regularization_strength, elastic_eta=elastic_eta)
                loss += reg_loss

    return predict_noise, loss, tape

# 保存训练数据
def generate_random_excel_file(output_dir='./result', num_digits=4):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 生成一个随机的文件名
    random_chars = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8 - num_digits))
    random_digits = ''.join(random.choice(string.digits) for _ in range(num_digits))
    random_filename = random_chars + random_digits + '.xlsx'

    # 构建完整的文件路径
    excel_path = os.path.join(output_dir, random_filename)

    # 检查是否存在.xlsx文件，如果存在则删除
    for filename in os.listdir(output_dir):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(output_dir, filename)
            os.remove(file_path)
    return excel_path

def print_sp(message):
    # Define ANSI escape codes for special font and color
    special_font = "\033[1m"  # Bold
    special_color = "\033[91m"  # Red

    # Reset text attributes to default after the message
    reset_attributes = "\033[0m"

    # Print the message with the special font and color
    print(f"{special_font}{special_color}\n{message}{reset_attributes}")

@tf.function
def reg(model, regularization, regularization_strength, elastic_eta):
    l1_norm = 0
    l2_norm = 0

    if regularization == "L1":
        for layer in model.layers:
            for weights in layer.trainable_weights:
                l1_norm += tf.reduce_sum(tf.abs(weights))
        return l1_norm * regularization_strength
    elif regularization == "L2":
        for layer in model.layers:
            for weights in layer.trainable_weights:
                l2_norm += tf.reduce_sum(tf.square(weights))
        return l2_norm * regularization_strength
    elif regularization == "L1&L2":
        for layer in model.layers:
            for weights in layer.trainable_weights:
                l1_norm += tf.reduce_sum(tf.abs(weights))
                l2_norm += tf.reduce_sum(tf.square(weights))
        return elastic_eta * l1_norm + (1 - elastic_eta) * l2_norm

# def reg(model, regularization,regularization_strength,elastic_eta):
#     l1_norm = 0
#     l2_norm = 0
#     for layer in model.layers:
#         for weights in layer.trainable_weights:
#             if regularization == "L1":
#                 l1_norm += tf.reduce_sum(tf.abs(weights))
#             elif regularization == "L2":
#                 l2_norm += tf.reduce_sum(tf.square(weights))
#             elif regularization == "L1&L2":
#                 l1_norm += tf.reduce_sum(tf.abs(weights))
#                 l2_norm += tf.reduce_sum(tf.square(weights))
#
#     if regularization == "L1":
#         return l1_norm * regularization_strength
#     elif regularization == "L2":
#         return l2_norm * regularization_strength
#     elif regularization == "L1&L2":
#         return elastic_eta * l1_norm + (1 - elastic_eta) * l2_norm
