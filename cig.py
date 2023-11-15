import os
from arg import arg

# 方便调试

'''-----init_cig------'''
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # 在tf的低版中需要手动开启xla
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少tf中的不必要的通知

args = arg.easy_arg()

'''-----PATH------'''
IMAGE_PATH = args.image_path
LABEL_PATH = args.label_path
TEST_PATH = args.test_path
LOG_PATH = args.log_path
RESULT_PATH = args.result_path
MODEL_SAVE = args.model_save
KEEP_ARG = args.keep_arg
'''-----Public------'''
LR = args.lr  # 固定学习率
LR_MAX = args.lr_max # 余弦退火的上限
LR_MIN = args.lr_min  # 余弦退火的下线
Cosine_Annealing = args.cosine_annealing # 选择True开启使用学习率的衰减
EPOCHS = args.epochs  # 总共训练的epoch
BACTH_SIZE = args.batch_size  # 训练时候的batch
MIXED_FLOAT16 = args.mixed_float16 # 选择True开启混合精度计算，注意如果开启混合精度计算，将带来精度的下降，以及梯度的消失
GBATCH_SIZE = args.gbatch_size  # 无使用
NUN_PROCESSES = args.num_processes  # 加载图片使用的工作器
LOSS_TYPE = args.loss_type  # 使用的损失函数,可选择L1、L2这两个指标
SCHEDELE = args.schedule  # “linear”
SCHEDELE_LOW = args.schedule_low # cosine 的上界
SCHEDELE_HIGH = args.schedule_high  # cosine 的下界
T = args.T  # 加入噪声的次数
LOAD_WEIGHT = args.load_weight  # 是否使用当地模型参数
SAVE_MODEL_COUNT = args.save_model_count  # 每训练100次保存模型
TEST_SAMPLE = args.test_sample  # test的时候产生的样本
TEST_COUNT = args.test_count  # 每n个epoch生成sample个图片来查看质量(这里建议给大一点，因为生成的样本的速度是很慢的)
if MIXED_FLOAT16 is True:
    EMA = False  # 是否使用ema ，注意如果开启混合精度训练，权重ema将无法使用，因为精度的问题
else:
    EMA = args.EMA
EMA_DECAY = args.EMA_decay
CLIP = args.clip  # 是否使用梯度截断
CLIP_VALUE = args.clip_value  # 梯度截断的阈值
REGULARIZATION = args.regularization  # you can choice L1 , L2 or L1&L2 ,default L2 , don't ues regularization you should use None
REGULARIZATION_STRENGTH = args.regularization_strength  # 正则化强度
ELASTIC_ETA = args.elastic_eta

'''-----DDIM------'''
MODEL_SAMPLE = args.model_sample  # default DDIP, DDMP you can select.
DDIM_SAMPLE_TIMES = args.DDIM_sample_times  # you should choose DDIP ,噪声次数是这个值的整数倍
DDIM_ETA = args.DDIM_ETA  # default 0.0 这是一个默认的数据，等于0，就是DDIM

'''-----NetCig------'''
ACTIVATE = args.activate  # 可用的激活函数，默认选择的是relu 在main文件中更改这个
DROPOUT_RATE = args.dropout_rate  # Unet中模块的dropout的大小
BN_NUM_BATCH = args.BN_num_batch  # GBN中的groups的参数
USE_ATTENTION_DOWN = args.use_attention_down#[False, False, True, True, False, False, True, True]  # 在unet左边模块中是否使用attention
SELF_ATTENTION_DOWN = args.self_attention_down#[False, False, False, False, False, False, False, False]  # 在unet左边模块是否使用self_attention
USE_ATTENTION_MID = args.use_attention_mid#[True, True]  # 在unet中间部分中是否使用attention
SELF_ATTENTION_MID = args.self_attention_mid#[False, False]  # 在unet中间部分中是否使用self_attention
USE_ATTENTION_UP = args.use_attention_up#[True, True, False, False, True, True, False, False]  # 在unet右边模块中是否使用attention
SELF_ATTENTION_UP =args.self_attention_up #[False, False, False, False, False, False, False, False]  # 在unet右边模块中是否使用self_attention

'''-----Adam------'''
OPT = args.optimizer
EPS = args.EPS
BETA_1 = args.BETA_1
BETA_2 = args.BETA_2

'''-----channels------'''
TIMES_CHANNELS = args.TIMES_CHANNELS  # 对加入噪声随机次数的编码后映射的维度
UNET_CHANNELS_DOWN = args.UNET_CHANNELS_DOWN  # Unet左边的参数
UNET_CHANNELS_MID = args.UNET_CHANNELS_MID # Unet中间的参数
UNET_CHANNELS_UP = args.UNET_CHANNELS_UP## Unet右边的参数
INICONV_CHANNELS = args.INICONV_CHANNELS # 一般和Unet左边第一个参数保持一致
TIME_EMB_CHANNELS = args.TIME_EMB_CHANNELS# 次数编码维度(times,time_emb_channels)编码后的数据长度，使用的正余弦编码

'''-----Image_deal------'''
PATCH_SIZE = args.PATCH_SIZE  # 图片分解为一个小的patch后的大小
IMAGE_INPUT_SIZE = args.IMAGE_INPUT_SIZE # 图片的输入的大小

'''-----Train------'''
LOSS = []  # 未使用
VAL_LOSS = []  # 未使用
