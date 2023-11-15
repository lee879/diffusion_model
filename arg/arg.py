import argparse
import os
from datetime import datetime

def easy_arg():
    parser = argparse.ArgumentParser()
    '''-----PATH------'''
    parser.add_argument("--image-path", type=str, default=r"./data/train/train_data/anime_faces/*.png",         help="Path to image data")
    parser.add_argument("--label-path", type=str, default=r"./data/train/train_label",                          help="Path to label data")
    parser.add_argument("--test-path", type=str, default=r"./test",                                             help="Path for testing")
    parser.add_argument("--log-path", type=str, default=r"./log",                                               help="Path for log files")
    parser.add_argument("--result-path", type=str, default=r"./result",                                         help="Path for results")
    parser.add_argument("--model-save", type=str, default=r"./model",                                           help="Path for saving models")
    parser.add_argument("--keep-arg", type=str,default=r"./arg/hyp" ,                                           help="keep hpy arg")

    '''-----Public------'''
    parser.add_argument("--lr", type=float, default=2e-4,                                                       help="Fixed learning rate")
    parser.add_argument("--lr-max", type=float, default=0.0002,                                                 help="Cosine annealing upper limit")
    parser.add_argument("--lr-min", type=float, default=2e-5,                                                   help="Cosine annealing lower limit")
    parser.add_argument("--cosine-annealing", type=bool, default=True,                                          help="Use cosine annealing for learning rate decay")
    parser.add_argument("--epochs", type=int, default=300,                                                     help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=64,                                                  help="Batch size for training")
    parser.add_argument("--mixed-float16", type=bool, default=False,                                             help="Use mixed precision computation")
    parser.add_argument("--gbatch-size", type=int, default=10,                                                  help="Unused parameter")
    parser.add_argument("--num-processes", type=int, default=4,                                                 help="Number of workers for loading images")
    parser.add_argument("--loss-type", type=str, default="L2",                                                  help="Type of loss function (L1, L2)")
    parser.add_argument("--schedule", choices=["cosine", "linear"], default="cosine",                           help="Learning rate schedule type")
    parser.add_argument("--schedule-low", type=float, default=1e-4,                                             help="Schedule low value")
    parser.add_argument("--schedule-high", type=float, default=0.02,                                            help="Schedule high value")
    parser.add_argument("--T", type=int, default=1000,                                                          help="Number of times to add noise")
    parser.add_argument("--load-weight", type=bool, default=False,                                              help="Use local model parameters")
    parser.add_argument("--save-model-count", type=int, default=1000,                                           help="Save model every N training steps")
    parser.add_argument("--test-sample", type=int, default=16,                                                  help="Number of samples for testing")
    parser.add_argument("--test-count", type=int, default=10,                                                   help="Generate N samples every M epochs for quality check")
    parser.add_argument("--EMA", type=bool, default=True,                                                       help="Use EMA")
    parser.add_argument("--EMA-decay", type=float, default=0.999,                                               help="EMA decay rate")
    parser.add_argument("--clip", type=bool, default=True,                                                      help="Use gradient clipping")
    parser.add_argument("--clip-value", type=float, default=0.5,                                                help="Gradient clipping threshold")
    parser.add_argument("--regularization", choices=["L1", "L2", "L1&L2", None], default=None,                  help="Type of regularization")
    parser.add_argument("--regularization-strength", type=float, default=1e-5,                                  help="Regularization strength")
    parser.add_argument("--elastic-eta", type=float, default=0.5,                                               help="Elastic eta")

    '''-----DDIM------'''
    parser.add_argument("--model-sample", choices=["DDIP", "DDMP"], default="DDIP",                             help="Model sample type")
    parser.add_argument("--DDIM-sample-times", type=int, default=50,                                            help="Number of DDIM sample times (must be a multiple of this value)")
    parser.add_argument("--DDIM-ETA", type=float, default=0.0,                                                  help="DDIM ETA")

    '''-----NetCig------'''
    parser.add_argument("--activate", choices=["relu", "swich", "tanh"], default="swich",                        help="Activation function for NetCig")
    parser.add_argument("--dropout-rate", type=float, default=0.0,                                              help="Dropout rate for Unet modules")
    parser.add_argument("--BN-num-batch", type=int, default=4,                                                  help="Number of groups for GBN")
    parser.add_argument("--use-attention-down", nargs='+', type=bool,
                        default=[False, False, True, True, False, False, True, True],
                                                                                                                help="Use attention in Unet left modules")
    parser.add_argument("--self-attention-down", nargs='+', type=bool,
                        default=[False, False, False, False, False, False, False, False],
                                                                                                                help="Use self-attention in Unet left modules")
    parser.add_argument("--use-attention-mid", nargs='+', type=bool, default=[False, False],
                                                                                                                help="Use attention in Unet middle modules")
    parser.add_argument("--self-attention-mid", nargs='+', type=bool, default=[False, False],
                                                                                                                help="Use self-attention in Unet middle modules")
    parser.add_argument("--use-attention-up", nargs='+', type=bool,
                        default=[True, True, False, False, True, True, False, False],
                                                                                                                help="Use attention in Unet right modules")
    parser.add_argument("--self-attention-up", nargs='+', type=bool,
                        default=[False, False, False, False, False, False, False, False],
                                                                                                                help="Use self-attention in Unet right modules")
    '''-----Adam------'''
    parser.add_argument("--optimizer", choices=["Adam"], default="Adam",                                        help="Optimizer type")
    parser.add_argument("--EPS", type=float, default=1e-8,                                                      help="Epsilon for Adam optimizer")
    parser.add_argument("--BETA_1", type=float, default=0.9,                                                   help="Beta 1 for Adam optimizer")
    parser.add_argument("--BETA_2", type=float, default=0.99,                                                  help="Beta 2 for Adam optimizer")

    '''-----channels------'''
    parser.add_argument("--TIMES-CHANNELS", type=int, default=256,                                               help="Dimension for encoding random times noise")
    parser.add_argument("--UNET-CHANNELS-DOWN", nargs='+', type=int, default=[32, 32, 64, 64, 128, 128, 256, 256],
                                                                                                                help="Parameters for Unet left side")
    parser.add_argument("--UNET-CHANNELS-MID", nargs='+', type=int, default=[512, 512],
                                                                                                                help="Parameters for Unet middle")
    parser.add_argument("--UNET-CHANNELS-UP", nargs='+', type=int, default=[256, 256, 128, 128, 64, 64, 32, 32],
                                                                                                                help="Parameters for Unet right side")
    parser.add_argument("--INICONV-CHANNELS", type=int, default=32,
                                                                                                                help="Consistent with the first parameter on the left side of Unet")
    parser.add_argument("--TIME-EMB-CHANNELS", type=int, default=128,
                                                                                                                help="Dimension after encoding (times, time_emb_channels), using sine and cosine encoding")

    '''-----Image_deal------'''
    parser.add_argument("--PATCH-SIZE", type=int, default=64,                                                   help="Size after decomposing images into small patches")
    parser.add_argument("--IMAGE-INPUT-SIZE", type=int, default=64,                                             help="Input size of images")

    return parser.parse_args()


def args_infom(hy_path):
    args = easy_arg()

    data_dict = {}

    for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
        data_dict[k] = v
    print("----------------------------------All Parser----------------------------------")
    for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print(f'\t{k}: {v}')
    print("----------------------------------end----------------------------------")

    # Find the next available index for the filename
    timestamp = datetime.now().strftime("%Y--%m--%d--%H--%M--%S--%f")
    index = 1
    while True:
        file_name = f"{index:03d}.txt"
        file_path = os.path.join(hy_path, file_name)

        if not os.path.exists(file_path):
            break
        index += 1

    # Save args to the text file
    with open(file_path, "w") as file:
        file.write("\t" + timestamp + "\n")
        for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
            file.write(f"\t{k}: {v}\n")

if __name__ == '__main__':
    args_infom(hy_path=r"D:\pj\diffusion_tf\arg\hyp")
