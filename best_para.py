import os
import cig
import glob
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from net.ddpm import DDPM
from net.ddim import DDIM
from util import ema_copy
from util.utils import time_print
from dataset.dataset import make_dataset
from loss.loss_fd import CosineAnnealingSchedule
from util.train_utils import get_betas, gd_record, mixed_pre, opt, generate_random_excel_file, print_sp
import optuna

# 定义目标函数，根据超参数进行模型训练并返回损失
def objective(trial):
    # 获取 Optuna 提供的超参数建议
    LR = trial.suggest_loguniform('LR', 1e-6, 1e-2)
    LR_MAX = trial.suggest_loguniform('LR_MAX', 1e-6, 1e-2)
    LR_MIN = trial.suggest_loguniform('LR_MIN', 1e-6, 1e-2)
    Cosine_Annealing = trial.suggest_categorical('Cosine_Annealing', [True, False])
    EPOCHS = trial.suggest_int('EPOCHS', 100, 1000)
    BACTH_SIZE = trial.suggest_categorical('BACTH_SIZE', [64, 128, 256])
    # 添加其他超参数的建议

    # 在这里将超参数设置为模型和训练中
    cig.LR = LR
    cig.LR_MAX = LR_MAX
    cig.LR_MIN = LR_MIN
    cig.Cosine_Annealing = Cosine_Annealing
    cig.EPOCHS = EPOCHS
    cig.BACTH_SIZE = BACTH_SIZE
    # 添加其他超参数的设置

    # 执行模型训练
    mixed_pre(cig.MIXED_FLOAT16)  # 设置混合精度训练
    img_path = glob.glob(cig.IMAGE_PATH)
    dataset, img_shape, data_len = make_dataset(img_path, cig.BACTH_SIZE, shuffle=cig.BACTH_SIZE * 8, resize=cig.IMAGR_INPUT_SIZE)
    # 添加数据加载和训练循环
    # 计算损失
    # 返回损失值作为目标函数的值

    return None  # 替换为实际的损失值

# 创建 Optuna Study 对象
study = optuna.create_study(direction='minimize')  # 'minimize' 用于最小化损失

# 运行超参数优化，设置适当的 n_trials（迭代次数）
study.optimize(objective, n_trials=100)

# 获取最佳超参数配置
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# 使用最佳超参数配置重新训练模型
# 替换原来的固定超参数值为 best_params

# 保存最终的模型和训练记录
