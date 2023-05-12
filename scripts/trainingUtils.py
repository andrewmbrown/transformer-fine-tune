# --- General Purpose Imports ---
import os
import yaml
import time
import wandb
import random
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# --- PyTorch Imports ---
import torch

# --- Memory Management Imports ---
from pynvml import *
import nvidia_smi


def load_config(args):
    """
    DESC:   Load yaml config file for training params
    INPUT:  args (argparse.ArgumentParser)
    OUTPUT: config_dict (dict) dictionary containing yaml file
    """
    assert args.config is not None, "ERROR: Please provide a path to yaml config file"
    # open yaml config as a strema and load into config_dict
    with open(args.config, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Configuration load failed!")
            print(exc)
    return config_dict


def set_seed(args, config_dict):
    """
    DESC:   Given seed value from args, set all the various pseudorandom seed
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing yaml file
    OUTPUT: None
    """
    assert config_dict['hyperparameters']['seed'] is not None, "ERROR: Please provide a seed value"
    random.seed(config_dict['hyperparameters']['seed'])
    np.random.seed(config_dict['hyperparameters']['seed'])
    torch.manual_seed(config_dict['hyperparameters']['seed'])
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(config_dict['hyperparameters']['seed'])
    print(f"Set all seeds to {config_dict['hyperparameters']['seed']}")
    return


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def check_cuda(args):
    """
    DESC: Check if CUDA is available for GPU training
    INPUT: args (argparse.ArgumentParser)
    OUTPUT: is_cuda (bool) flag to indicate if CUDA is available
    """
    if torch.cuda.is_available():
        is_cuda = True
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.cuda = True
        print(f"CUDA is available. {args.n_gpu} GPU(s) will be used.")
    else:
        is_cuda = False
        args.device = torch.device("cpu")
        args.n_gpu = 0
        args.cuda = False
        print("CUDA is not available. CPU will be used.")
    return is_cuda

def print_gpu_utilization():
    """
    DESC: Check gpu utilization, currently for only one device
    INPUT: None
    OUTPUT: None (prints gpu diagnostic)
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def init_wandb(args, config_dict):
    """
    DESC:   Initialize wandb for logging
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing yaml file
    OUTPUT: None
    """
    # load key from os environment variable
    key = os.environ.get("WANDB_KEY")
    if args.use_wandb:
        try:
            wandb.login(key=key)
            wandb.init(
                project=config_dict["wandb_project_name"],
                notes=config_dict["wandb_notes"],
                config=config_dict,
                tags=config_dict["wandb_tags"])
            print("wanbd login and init successful, logging run!")
        except:
            print("WARNING: wandb key provided is invalid. Wandb will not be used.")
    else:
        print("WARNING: No wandb key provided. Wandb will not be used.")
    return


def main():
    pass

# if name main prevents code from running when imported
if __name__ == "__main__":
    main()