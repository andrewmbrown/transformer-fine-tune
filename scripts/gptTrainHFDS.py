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
import deepspeed
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
from datasets import load_dataset

# Memory Management Imports
from pynvml import *
import nvidia_smi

# --- Custom File Imports ---
from gpt2Dataset import GPT2Dataset
from trainingUtils import *


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


def load_raw_train_validation_datasets(args, config_dict):
    """
    DESC:   Load train and validation data as raw csv files
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing yaml file
    OUTPUT: raw_datasets (dict) dictionary containing train and validation datasets
    """
    assert config_dict['data_train_path'] is not None, "ERROR: Please provide a path to training data"
    assert config_dict['data_validation_path'] is not None, "ERROR: Please provide a path to validation data"
    raw_datasets = load_dataset('csv', data_files=config_dict['data_train_path'])
    raw_datasets["validation"] = (load_dataset('csv', data_files=config_dict['data_validation_path']))["train"]
    return raw_datasets


def init_tokenizer(args, config_dict):
    """
    DESC:   Initialize GPT2 tokenizer (byte-level BPE tokenizer)
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
    OUTPUT: tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
    """
    assert config_dict['model_name'] is not None, "ERROR: Please provide a tokenizer name"
    # model name is the same as tokenizer, so we can use it to load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config_dict['model_name'])
    # set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_datasets(args, config_dict, tokenizer, raw_datasets):
    """
    DESC:   Tokenize datasets using GPT2 tokenizer
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
            raw_datasets (dict) dictionary containing train and validation datasets
    OUTPUT: tokenized_datasets (dict) dictionary containing tokenized train and validation datasets
    """
    # helper for tokenizing mapping tokenization function to each example
    def tokenize_function(examples):
        return tokenizer(examples["triplet"], truncation=True)
    
    # tokenize datasets
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # remove "triplet" text column from dataset
    tokenized_datasets = tokenized_datasets.remove_columns(["triplet"])
    return tokenized_datasets


def init_model(args, config_dict, tokenizer):
    """
    DESC:   Load gpt2 configuration, then model
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
    OUTPUT: model (GPT2LMHeadModel) GPT2 model
    """
    assert config_dict['model_name'] is not None, "ERROR: Please provide a model name"
    # Load GPT2 configuration
    config = GPT2Config.from_pretrained(config_dict['model_name'], output_hidden_states=False)
    # instantiate model
    model = GPT2LMHeadModel.from_pretrained(config_dict['model_name'], config=config)
    # resize token embeddings for our custom tokens
    model.resize_token_embeddings(len(tokenizer))
    # move model to GPU (if available)
    model.to(args.device)
    return model


def load_optimizer(args, config_dict, model):
    """
    DESC:   Load optimizer (currently only AdamW) TODO: add other optimizers
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            model (GPT2LMHeadModel) GPT2 model
    OUTPUT: optimizer (AdamW) AdamW optimizer
    """
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    optimizer = AdamW(model.parameters(),
                    lr = config_dict['learning_rate'],
                    eps = config_dict['epsilon']
                    )
    return optimizer


def init_scheduler(args, config_dict, optimizer, train_dataloader):
    """
    DESC:   Given optimizer and dataloader, init scheduler with warmup steps
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            optimizer (torch.optim.xxx or transformers.xxx) optimizer
            train_dataloader (torch.DataLoader) dataloader used to get number of training steps
    OUTPUT: scheduler (get_linear_schedule_with_warmup) scheduler
    """
    # Scheduler uses warmup steps to avoid large learning rate at the beginning of training
    # We need the total number of training steps to calculate the learning rate at each step
    # Here we obtain it from the length of the dataloader and number of epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = config_dict['num_warmup_steps'],
                                            num_training_steps = len(train_dataloader)*config_dict['num_epochs'])
    return scheduler

def certify_training_config(args, config_dict):
    """
    DESC:   Check that all required training configs are present
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
    OUTPUT: None
    """
    print("Checking that all required training configs are present...")
    assert config_dict['model_name'] is not None, "ERROR: Please provide a model name"
    assert config_dict['output_model_dir'] is not None, "ERROR: Please provide an output model directory"
    assert config_dict['hyperparameters']["learning_rate"] is not None, "ERROR: Please provide a learning rate"
    assert config_dict['hyperparameters']["epochs"] is not None, "ERROR: Please provide a number of epochs"
    assert config_dict['num_warmup_steps'] is not None, "ERROR: Please provide a number of warmup steps"
    assert config_dict['batch_size'] is not None, "ERROR: Please provide a batch size"
    if args.save_model == False:
        print("WARNING: You have not specified False to saving model. Your model will not be saved.")
    if args.save_tokenizer == False:
        print("WARNING: You have not specified False to saving tokenizer. Your tokenizer will not be saved.")
    if args.save_training_args == False:
        print("WARNING: You have not specified False to saving training args. Your training args will not be saved.")
    print("All required training configs are present!")
    

def init_trainer(args, config_dict, model, tokenizer, tokenized_datasets):
    """
    DESC:   Initialize trainer object
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            model (GPT2LMHeadModel) untrained GPT2 model
            tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
            tokenized_datasets (dict) dictionary containing tokenized train and validation datasets
    OUTPUT: trainer (transformers.Trainer) trainer object
    """
    # first check and validate that all required configs are present
    certify_training_config(args, config_dict)
    # define training args
    training_args = TrainingArguments(output_dir=config_dict['output_model_dir'],
                                    overwrite_output_dir=True,
                                    deepspeed=args.path_to_deepspeed_config,
                                    evaluation_strategy = "steps", # used to be epoch
                                    prediction_loss_only = True, #get rid of this if we end up adding metrics
                                    logging_dir=f"./logs/",
                                    logging_strategy="steps",
                                    logging_steps=5,
                                    save_strategy="no",
                                    bf16=True,
                                    num_train_epochs=config_dict['hyperparameters']["epochs"],
                                    per_device_train_batch_size=config_dict['hyperparameters']["batch_size"],
                                    per_device_eval_batch_size=config_dict['hyperparameters']["eval_batch_size"],
                                    learning_rate=config_dict['hyperparameters']["learning_rate"],
                                    weight_decay=config_dict['hyperparameters']["weight_decay"],
                                    seed=config_dict['hyperparameters']["seed"],
                                    eval_steps=config_dict['hyperparameters']["eval_steps"]
                                    )
    # instantiate trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer
    )
    return trainer


def train_and_save_model(args, config_dict, trainer):
    """
    DESC:   Train model
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            trainer (transformers.Trainer) trainer object
    OUTPUT: None
    """
    # train model
    trainer.train()

    # depending on args, save model, tokenizer, training args, and training config
    if args.save_model:
        # save model
        trainer.save_model(config_dict['output_model_dir'])
    if args.save_tokenizer:
        # save tokenizer
        trainer.tokenizer.save_pretrained(config_dict['output_model_dir'])
    
    if args.save_training_args:
        # save training arguments
        trainer.save_args(config_dict['output_model_dir'])
        # save training config
        with open(os.path.join(config_dict['output_model_dir'], 'training_config.yaml'), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    return



def main(args):
    # GPT2 Fine-tuning pipeline
    # --- check cuda availability
    check_cuda(args)
    # --- Load yaml using args.config
    config_dict = load_config(args)
    # --- Load raw datasets from CSVs
    raw_datasets = load_raw_train_validation_datasets(args, config_dict)
    # --- Initialize tokenizer
    tokenizer = init_tokenizer(args, config_dict)
    # tokenize datasets
    tokenized_datasets = tokenize_datasets(args, config_dict, tokenizer, raw_datasets)
    # --- Initialize model
    model = init_model(args, config_dict, tokenizer)
    # --- Initialize optimizer
    """ Optimizer and scheduler are currently defaulty handled by the Trainer object - no need
    optimizer = load_optimizer(args, config_dict, model)
    # --- Initialize scheduler
    scheduler = init_scheduler(args, config_dict, optimizer)
    """
    # --- Initialize dataloaders
    """ Dataloaders are currently defaulty handled by the Trainer object - no need
    train_dataloader = init_dataloader(args, config_dict, tokenizer, train_df, mode="Random")
    val_dataloader = init_dataloader(args, config_dict, tokenizer, train_df, mode="Sequential")
    """
    # --- Initialize trainer
    trainer = init_trainer(args, config_dict, model, tokenizer, tokenized_datasets)
    # --- Initialize wandb
    if args.use_wandb:
        init_wandb(args, config_dict)
    else:
        print("Not logging to wandb!")

    # --- Set Seed
    set_seed(args, config_dict)
    # --- Train model and save
    train_and_save_model(args, config_dict, trainer)


if __name__ == "__main__":
    # --- Instantiate Argument Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to configuration YAML file", required=True)
    parser.add_argument("-t", "--path_to_train_data", help="Path to training data", required=False)
    parser.add_argument("-v", "--path_to_val_data", help="Path to validation data", required=False)
    parser.add_argument("-w", "--path_to_wandb_key", help="Path to wandb login key", required=False)

    global args  # set global args scope potential for gpu diagnostics
    args = argparse.parse_args()
    main(args)
