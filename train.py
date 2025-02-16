import os

import argparse

from torch.backends import cudnn
from config import config, dataset_config, merge_cfg_arg

from dataloder import get_loader
from solver_makeup import Solver_makeupGAN


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN")
    # general
    parser.add_argument(
        "--data_path",
        default="./mtdataset",
        type=str,
        help="training and test data path",
    )
    parser.add_argument("--dataset", default="MAKEUP", type=str)
    parser.add_argument(
        "--gpus", default="0", type=str, help="GPU device to train with"
    )
    parser.add_argument("--batch_size", default="1", type=int, help="batch_size")
    parser.add_argument(
        "--vis_step", default="1260", type=int, help="steps between visualization"
    )
    parser.add_argument(
        "--task_name", default="makeup_branch_1", type=str, help="task name"
    )
    parser.add_argument("--checkpoint", default="", type=str, help="checkpoint to load")
    parser.add_argument(
        "--ndis", default="1", type=int, help="train discriminator steps"
    )
    parser.add_argument("--LR", default="2e-4", type=float, help="Learning rate")
    parser.add_argument(
        "--decay", default="0", type=int, help="epochs number for training"
    )
    parser.add_argument(
        "--model",
        default="makeupGAN",
        type=str,
        help="which model to use: cycleGAN/ makeupGAN",
    )
    parser.add_argument("--epochs", default="300", type=int, help="nums of epochs")
    parser.add_argument(
        "--whichG",
        default="branch",
        type=str,
        help="which Generator to choose, normal/branch, branch means two input branches",
    )
    parser.add_argument(
        "--norm",
        default="SN",
        type=str,
        help="normalization of discriminator, SN means spectrum normalization, none means no normalization",
    )
    parser.add_argument(
        "--d_repeat",
        default="3",
        type=int,
        help="the repeat Res-block in discriminator",
    )
    parser.add_argument(
        "--g_repeat", default="6", type=int, help="the repeat Res-block in Generator"
    )
    parser.add_argument(
        "--lambda_cls", default="1", type=float, help="the lambda_cls weight"
    )
    parser.add_argument(
        "--lambda_rec", default="10", type=int, help="lambda_A and lambda_B"
    )
    parser.add_argument(
        "--lambda_his", default="1", type=float, help="histogram loss on lips"
    )
    parser.add_argument(
        "--lambda_skin_1",
        default="0.1",
        type=float,
        help="histogram loss on skin equals to lambda_his* lambda_skin",
    )
    parser.add_argument(
        "--lambda_skin_2",
        default="0.1",
        type=float,
        help="histogram loss on skin equals to lambda_his* lambda_skin",
    )
    parser.add_argument(
        "--lambda_eye",
        default="1",
        type=float,
        help="histogram loss on eyes equals to lambda_his*lambda_eye",
    )
    parser.add_argument(
        "--content_layer",
        default="r41",
        type=str,
        help="vgg layer we use to output features",
    )
    parser.add_argument(
        "--lambda_vgg", default="5e-3", type=float, help="the param of vgg loss"
    )
    parser.add_argument(
        "--cls_list", default="N,M", type=str, help="the classes of makeup to train"
    )
    parser.add_argument(
        "--direct",
        type=bool,
        default=True,
        help="direct means to add local cosmetic loss at the first, unified training",
    )
    parser.add_argument(
        "--lips", type=bool, default=True, help="whether to finetune lips color"
    )
    parser.add_argument(
        "--skin", type=bool, default=True, help="whether to finetune foundation color"
    )
    parser.add_argument(
        "--eye", type=bool, default=True, help="whether to finetune eye shadow color"
    )
    args = parser.parse_args()
    return args


def train_net():
    # enable cudnn
    cudnn.benchmark = True

    data_loaders = get_loader(config, mode="train")  # return train&test
    # get the solver
    # if args.model == "cycleGAN":
    #     solver = Solver_makeupGAN(data_loaders, config, dataset_config)
    # solver = Solver_cycleGAN(data_loaders, config, dataset_config)
    if args.model == "makeupGAN":
        solver = Solver_makeupGAN(data_loaders, config, dataset_config)
    else:
        print("model that not support")
        exit()
    solver.train()


if __name__ == "__main__":
    args = parse_args()
    config = merge_cfg_arg(config, args)

    dataset_config.name = args.dataset
    print("show config")
    for arg in vars(config):
        print(f"{str(arg):>15}: {str(getattr(config, arg)):>30}")
    print()

    # Create the directories if not exist
    if not os.path.exists(config.data_path):
        print(f"No data path!! ({config.data_path})")
        exit()

    train_net()
