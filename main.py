import os
import cig
import glob
import time
from arg import arg
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
from util.train_utils import get_betas, gd_record, mixed_pre, opt, generate_random_excel_file,print_sp
arg.args_infom(hy_path=cig.KEEP_ARG)
# 计算时间
start_time = time.time()

# 是否使用混合精度训练
mixed_pre(cig.MIXED_FLOAT16)

# 加载数据,只考虑训练loss
img_path = glob.glob(cig.IMAGE_PATH)
dataset, img_shape, data_len = make_dataset(img_path, cig.BACTH_SIZE, shuffle=cig.BACTH_SIZE * 8,
                                            resize=cig.IMAGE_INPUT_SIZE)
dataset = dataset.repeat()
db_iter = iter(dataset)

# 创建tensorboard和model保存的路径
summary_writer = tf.summary.create_file_writer(cig.LOG_PATH)
checkpoint_dir = cig.MODEL_SAVE
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
best_weights_checkpoint_path_d = os.path.join(checkpoint_dir, 'best_d.hd5')  # 获得保存模型的路径

# 得到result文件保存路径(每一次都是随机生成的)
excel_path = generate_random_excel_file(output_dir=cig.RESULT_PATH)

# 得到betas
betas = get_betas(cig.SCHEDELE, cig.T)

# 扩散模型导入模型加载模型
if cig.MODEL_SAMPLE == "DDPM":
    model = DDPM(betas=betas,
                 image_size=cig.IMAGE_INPUT_SIZE,
                 channels=(cig.UNET_CHANNELS_DOWN, cig.UNET_CHANNELS_MID, cig.UNET_CHANNELS_DOWN),
                 times_encoder_channels=cig.TIMES_CHANNELS,
                 iniconv_channels=cig.INICONV_CHANNELS,
                 dropout_rate=cig.DROPOUT_RATE,
                 use_attention=(cig.USE_ATTENTION_DOWN, cig.USE_ATTENTION_MID, cig.USE_ATTENTION_UP),
                 self_attention=(cig.SELF_ATTENTION_DOWN, cig.SELF_ATTENTION_MID, cig.SELF_ATTENTION_UP),
                 num_batch=cig.BN_NUM_BATCH,
                 activate=cig.ACTIVATE,
                 T=cig.T,
                 )
else:
    model = DDIM(betas=betas,
                 image_size=cig.IMAGE_INPUT_SIZE,
                 channels=(cig.UNET_CHANNELS_DOWN, cig.UNET_CHANNELS_MID, cig.UNET_CHANNELS_DOWN),
                 times_encoder_channels=cig.TIMES_CHANNELS,
                 iniconv_channels=cig.INICONV_CHANNELS,
                 dropout_rate=cig.DROPOUT_RATE,
                 use_attention=(cig.USE_ATTENTION_DOWN, cig.USE_ATTENTION_MID, cig.USE_ATTENTION_UP),
                 self_attention=(cig.SELF_ATTENTION_DOWN, cig.SELF_ATTENTION_MID, cig.SELF_ATTENTION_UP),
                 num_batch=cig.BN_NUM_BATCH,
                 activate=cig.ACTIVATE,
                 T=cig.T,
                 time_emb_channels=cig.TIME_EMB_CHANNELS
                 )

    print_sp("\nuse DDIM model.......") # 特色输出

if cig.EMA: # 判断是否使用ema
    ema_model = None
    ema = ema_copy.EMA(decay=cig.EMA_DECAY)

# 是否导入本地模型权重
if cig.LOAD_WEIGHT:
    model.load_weights("./model/best_d.hd5")
    print("\nthe local model was successfully loaded......")
else:
    print("\nDo not use a lacal mosel......")

# 自定义训练部分
count = 0  # 计数器
training_info = []  # 保存每一次训练的参数
for epoch in range(cig.EPOCHS):
    train_loss_epoch = []
    # 是否使用余弦退火算法
    if cig.Cosine_Annealing:
        cos_lr = CosineAnnealingSchedule(lr_min=cig.LR_MIN, lr_max=cig.LR_MAX, T=cig.EPOCHS)
        lr = cos_lr(epoch)
    else:
        lr = cig.LR
    # 使用进度条
    pbar = tqdm(range(data_len), desc=f"Epoch {epoch}", position=0, leave=True)
    opt_ada = opt(select=cig.MIXED_FLOAT16, lr=lr, eps=cig.EPS, beta1=cig.BETA_1, beta2=cig.BETA_2) #初始化优化器
    for step in pbar:
        train_data = next(db_iter)
        predict_noise, loss, tape = gd_record(train_image=train_data,
                                              model=model,
                                              sample=None,
                                              test=False,
                                              select_loss=cig.LOSS,
                                              regularization=cig.REGULARIZATION,
                                              regularization_strength=cig.REGULARIZATION_STRENGTH,
                                              elastic_eta=cig.ELASTIC_ETA
                                              )
        gd = tape.gradient(loss, model.trainable_variables)
        if cig.CLIP:  # 是否使用梯度的截断
            for i, t in enumerate(gd):
                gd[i] = tf.clip_by_norm(t, clip_norm=cig.CLIP_VALUE)
        opt_ada.apply_gradients(zip(gd, model.trainable_variables))
        train_loss_epoch.append(loss)
        if cig.EMA:  # 使用ema
            if ema_model is None:
                ema_model = deepcopy(model)
            ema.update_model_average(ema_model, model)
        with summary_writer.as_default():
            tf.summary.scalar('loss_mes', loss, step=count)
            pbar.set_postfix({"Train_Loss": float(loss), "Count": count, "Lr": float(lr)})
            # 将结果部分训练结果保存在csv文件中
            training_info.append({
                "Epoch": epoch,
                "Step": count,
                "Loss": float(loss),
                "LearningRate": float(lr)
            })
            # 每经过一段时间保存模型
            if count % cig.SAVE_MODEL_COUNT == 0:
                model.save_weights(best_weights_checkpoint_path_d)  # 保存模型

                training_df = pd.DataFrame(training_info)
                if os.path.exists(excel_path):
                    existing_df = pd.read_excel(excel_path)
                    updated_df = pd.concat([existing_df, training_df], ignore_index=True)
                    updated_df.to_excel(excel_path, index=False)
                else:
                    training_df.to_excel(excel_path, index=False)
                training_info = []  # 保存完数据后应该，清空数据防止数据的叠加
                # 防止出现内存溢出
                tf.keras.backend.clear_session()
                print("\nTraining information saved up to epoch:", epoch)
            count += 1
    print("\n",epoch, "loss", float(np.mean(np.array(train_loss_epoch))))

    if epoch % cig.TEST_COUNT == 0:
        print("\n-----------------test_start---------------------------")
        print("\nwait a little time......")
        if cig.EMA:
            print("\nuse ema_model......")
            sample = ema_model(train_inputs=None, sample=cig.TEST_SAMPLE, test=True,training = False)
        else:
            sample = model(train_inputs=None, sample=cig.TEST_SAMPLE, test=True, training=False)

        with summary_writer.as_default():
            tf.summary.image("predict_0", (sample[:3, :, :, :] + 1) / 2, step=epoch)
            tf.summary.image("original", train_data[:3, :, :, :], step=epoch)
        print("\n-----------------test_over---------------------------")
# 获得本次训练的时间
end_time = time.time()
time_print((end_time - start_time))