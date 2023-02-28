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

# --- Custom File Imports ---
from gpt2Dataset import GPT2Dataset


def set_seed(args):
    """
    DESC:   Given seed value from args, set all the various pseudorandom seed
    INPUT:  args (argparse.ArgumentParser)
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
    DESC:   Check if CUDA is available for GPU training
    INPUT:  args (argparse.ArgumentParser)
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


def load_yaml(args):
    """
    DESC:   Load yaml config file for training params
    INPUT:  args (argparse.ArgumentParser)
    OUTPUT: config_dict (dict) dictionary containing yaml file
    """
    assert args.path_to_config is not None, "Please provide a path to yaml config file"
    # open yaml config as a strema and load into config_dict
    with open(args.path_to_config, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Configuration load failed!")
            print(exc)
    return config_dict


def init_wandb(args, config_dict):
    """
    DESC:   Initialize wandb for logging
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing yaml file
    OUTPUT: None
    """
    assert args.path_to_wandb_key is not None, "Please provide a path to wandb key"
    # load wandb key
    with open(args.path_to_wandb_key, "r") as stream:
        try:
            wandb_key = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Wandb key load failed!")
            print(exc)
    # initialize wandb
    wandb.login(key=wandb_key)
    wandb.init(
        project=config_dict["wandb_project_name"],
        name=config_dict["wandb_run_name"],
        notes=config_dict["wandb_notes"],
        config=config_dict,
        tags=config_dict["wandb_tags"])
    return


def load_train_and_validation_as_df(args, config_dict):
    """
    DESC:   Load train and validation data as pandas dataframes
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing yaml file
    OUTPUT: train_df (pd.DataFrame) dataframe containing training data
            val_df (pd.DataFrame) dataframe containing validation data
    """
    assert args.path_to_train_data is not None, "Please provide a path to training data"
    assert args.path_to_val_data is not None, "Please provide a path to validation data"
    train_df = pd.read_csv(config_dict["data_train_path"])
    val_df = pd.read_csv(config_dict["data_validation_path"])
    return train_df, val_df


def preprocess_df(df):
    """
    DESC:   Given df to remove NaNs and copy data over as df containing only plaintext
            This triplet df is used to make the dataset for training and validation
    INPUT:  df (pd.DataFrame) dataframe to be preprocessed
    OUTPUT: triplets (pd.DataFrame) preprocessed dataframe
    """
    # remove NaNs
    df = df.dropna(inplace=True)
    triplets = df.triplet.copy()  # copy over triplets
    return triplets


def load_tokenizer(args, config_dict):
    """
    DESC:   Load tokenizer
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
    OUTPUT: tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
    """
    assert args.path_to_tokenizer is not None, "Please provide a path to tokenizer"
    # model name is the same as tokenizer, so we can use it to load the tokenizer
    # bos_token, eos_token, and pad_token are custom and added to the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config_dict['model_name'], bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    return tokenizer


def load_dataset(args, config_dict, tokenizer, train_df, val_df):
    """
    DESC:   Load dataframes into GPT2Dataset objects (tokenized with len and getitem overrides)
            Then load into torch DataLoader objects (with sampler type and batch_size)
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
            train_df (pd.DataFrame) dataframe containing training data
            val_df (pd.DataFrame) dataframe containing validation data
    OUTPUT: train_dataset (GPT2Dataset) dataset containing training data
            val_dataset (GPT2Dataset) dataset containing validation data
    """
    assert config_dict['max_length'] is not None, "Please provide a max length for the dataset"
    assert tokenizer is not None, "Please provide a tokenizer" 
    # Create custom dataset objects with the training and validation data
    train_dataset = GPT2Dataset(train_df, tokenizer, config_dict['max_length'])
    val_dataset = GPT2Dataset(val_df, tokenizer, config_dict['max_length'])
    # Load datasets into torch DataLoader objects
    # take training samples in random order
    train_dataloader = DataLoader(train_dataset,
                                sampler=RandomSampler(train_dataset),
                                batch_size=config_dict['batch_size'])

    # For validation, the order doesn't matter, so we read sequentially
    validation_dataloader = DataLoader(val_dataset,
                                    sampler=SequentialSampler(val_dataset),
                                    batch_size=config_dict['batch_size'])
    return train_dataloader, validation_dataloader


def load_model(args, config_dict, tokenizer):
    """
    DESC:   Load gpt2 configuration, then model
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
    OUTPUT: model (GPT2LMHeadModel) GPT2 model
    """
    assert config_dict['model_name'] is not None, "Please provide a model name"
    # Load GPT2 configuration
    config = GPT2Config.from_pretrained(config_dict['model_name'], output_hidden_states=False)
    # instantiate model
    model = GPT2LMHeadModel.from_pretrained(config_dict['model_name'], config=config)
    # resize token embeddings for our custom tokens
    model.resize_token_embeddings(len(tokenizer))
    # move model to GPU (if available)
    model.to(args.device)
    return model


def main():
    # --- Instantiate Argument Parser ---
    # path_to_data, 
    parser = argparse.ArgumentParser()
    global args
    args = argparse.parse_args()



if __name__ == "__main__":
    main()