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


# --- Machine Learning Imports ---
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup


def set_seed(args):
    """
    DESC: Given seed value from args, set all the various pseudorandom seed
    INPUT: args (argparse.ArgumentParser)
    OUTPUT: None
    """
    assert args.seed is not None, "Please provide a seed value"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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


def main():
    pass

# if name main prevents code from running when imported
if __name__ == "__main__":
    main()