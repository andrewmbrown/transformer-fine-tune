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
import nltk
import torch
import evaluate  # hf evaluation library
import deepspeed
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import Trainer, TrainingArguments, AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from datasets import load_dataset
from datasets import Dataset

# Memory Management Imports
from pynvml import *
import nvidia_smi

# --- Custom File Imports ---
from trainingUtils import *


def load_raw_train_validation_datasets(args, config_dict):
    """
    DESC:   Load train and validation data as raw csv files
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing yaml file
    OUTPUT: raw_datasets (dict) dictionary containing train and validation datasets
    """
    assert config_dict['data_train_path'] is not None, "ERROR: Please provide a path to training data"
    assert config_dict['data_validation_path'] is not None, "ERROR: Please provide a path to validation data"
    df = pd.read_csv(config_dict["data_train_path"])
    df_val = pd.read_csv(config_dict["data_validation_path"])
    train_raw_datasets = Dataset.from_pandas(df)
    val_raw_datasets = Dataset.from_pandas(df_val)
    return train_raw_datasets, val_raw_datasets


def init_tokenizer(args, config_dict):
    """
    DESC:   Initialize LM tokenizer (byte-level BPE tokenizer)
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
    OUTPUT: tokenizer (AutoTokenizer) tokenizer for LM model
    """
    assert config_dict['model_name'] is not None, "ERROR: Please provide a tokenizer name"
    # model name is the same as tokenizer, so we can use it to load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config_dict['model_name'])
    # set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_datasets(args, config_dict, tokenizer, raw_datasets):
    """
    DESC:   Tokenize datasets using LM tokenizer
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            tokenizer (AutoTokenizer) tokenizer for LM model
            raw_datasets (dict) dictionary containing train and validation datasets
    OUTPUT: tokenized_datasets (dict) dictionary containing tokenized train and validation datasets
    """
    # helper for tokenizing mapping tokenization function to each example
    def tokenize_function(examples):
        return tokenizer(examples["data"], truncation=True)
    
    # tokenize datasets
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # remove "triplet" text column from dataset
    tokenized_datasets = tokenized_datasets.remove_columns(["data"])
    return tokenized_datasets


def init_model(args, config_dict, tokenizer):
    """
    DESC:   Load LM configuration, then model
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            tokenizer (AutoTokenizer) tokenizer for LM model
    OUTPUT: model (AutoModelForCausalLM) LM model
    """
    assert config_dict['model_name'] is not None, "ERROR: Please provide a model name"
    assert args.device is not None, "ERROR: Please provide a device (cpu/gpu/etc) to mount model"
    # Load LM configuration
    config = AutoConfig.from_pretrained(config_dict['model_name'], output_hidden_states=False)
    # instantiate model
    model = AutoModelForCausalLM.from_pretrained(config_dict['model_name'], config=config)
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
            model (AutoModelForCausalLM) LM model
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

    # asserts for necessary training config arguments
    assert config_dict['model_name'] is not None, "ERROR: Please provide a model name"
    assert config_dict['output_model_dir'] is not None, "ERROR: Please provide an output model directory"
    assert config_dict['hyperparameters']["learning_rate"] is not None, "ERROR: Please provide a learning rate"
    assert config_dict['hyperparameters']["epochs"] is not None, "ERROR: Please provide a number of epochs"
    assert config_dict['hyperparameters']['warmup_steps'] is not None, "ERROR: Please provide a number of warmup steps"
    assert config_dict['hyperparameters']['batch_size'] is not None, "ERROR: Please provide a batch size"

    # argparser flags for training saving options
    # argparse parses arguments as string literals, so we cannot treat as booleans
    if args.save_model:
        print(f"Saving model to: {os.path.abspath(config_dict['output_model_dir'])}")
    else:
        print("WARNING: You have specified False to saving model. Your model will not be saved.")
    if args.save_tokenizer:
        print(f"Saving tokenizer to: {os.path.abspath(config_dict['output_tokenizer_dir'])}")
    else:
        print("WARNING: You have specified False to saving tokenizer. Your tokenizer will not be saved.")
    if args.save_arguments:
        print(f"Saving training arguments and config to: {os.path.abspath(config_dict['output_training_args_dir'])}")
    else:
        print("WARNING: You have specified False to saving training args. Your training args will not be saved.")

    # certify early stopping criteria
    if config_dict['use_early_stopping']:
        assert config_dict['load_best_model_at_end'] is not None
        assert config_dict['metric_for_best_model'] is not None
        assert config_dict['early_stopping_patience'] is not None
        assert config_dict['greater_is_better'] is not None
        print(f"WARNING: You have specified to use EARLY STOPPING on metric: {config_dict['metric_for_best_model']} with greater is better set to: {config_dict['greater_is_better']}")
        print(f"WARNING: EARLY STOPPING patience: {config_dict['early_stopping_patience']}")

    if args.use_deepspeed:
        assert config_dict['deepspeed_config_path'] is not None
        print(f"WARNING: You have specified to use Deepspeed, below is config path")
        print(f"{config_dict['deepspeed_config_path']}")
    else:
        print("WARNING: Not using deepspeed")

    print("All required training configs are present!")
    return
    

def compute_metrics(eval_preds):
    """
    DESC:   During training, compute automated metrics for evaluation
            Currently using rougeLSum
    INPUT:  eval_preds (tuple) tuple containing predictions and labels
    OUTPUT: result (dict) dictionary containing metrics
    """
    # unpack eval_preds tuple into predictions and labels
    preds, labels = eval_preds

    # instantiate rougeLSum metric
    # Setup evaluation
    nltk.download("punkt", quiet=True)
    metric = evaluate.load("rouge")
    # bertscore = evaluate.load("bertscore")

    # tokenizer doesn't exist in this scope, so we need to initialize it
    local_dict = {"model_name": "LM"}
    tokenizer = init_tokenizer(0, local_dict)    

    # using tokenizer, decode predictions and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Now we can decide on which metrics to use for evaluation
    # bertscore uses embediddings to compute semantic similarity
    # results = bertscore.compute(predictions=decoded_preds, references=decoded_labels, model_type="distilbert-base-uncased")

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result


def init_trainer(args, config_dict, model, tokenizer, train_tokenized_dataset, val_tokenized_dataset):
    """
    DESC:   Initialize trainer object
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            model (AutoModelForCausalLM) untrained AutoModelForCausalLM model
            tokenizer (AutoTokenizer) tokenizer for LM model
            tokenized_datasets (dict) dictionary containing tokenized train and validation datasets
    OUTPUT: trainer (transformers.Trainer) trainer object
    """
    # first check and validate that all required configs are present
    certify_training_config(args, config_dict)
    # define training args

    # some extra logic for giving deepspeed json config - probably temporary
    if args.use_deepspeed:
        training_args = Seq2SeqTrainingArguments(output_dir=config_dict['output_model_dir'],
                                        overwrite_output_dir=True,
                                        evaluation_strategy=config_dict['evaluation_strategy'],
                                        eval_steps=config_dict["eval_steps"],
                                        eval_accumulation_steps=config_dict["eval_accumulation_steps"],
                                        deepspeed=config_dict['deepspeed_config_path'],
                                        logging_dir=config_dict['logging_dir'],
                                        logging_strategy=config_dict['logging_strategy'],
                                        logging_steps=config_dict['logging_steps'],
                                        predict_with_generate=config_dict['predict_with_generate'],
                                        generation_max_length=config_dict['generation_max_length'],
                                        save_total_limit=config_dict['save_total_limit'],
                                        metric_for_best_model=config_dict['metric_for_best_model'],
                                        load_best_model_at_end=config_dict['load_best_model_at_end'],
                                        per_device_eval_batch_size=config_dict["eval_batch_size"],
                                        per_device_train_batch_size=config_dict['hyperparameters']["batch_size"],
                                        bf16=config_dict['hyperparameters']['bf16'],
                                        num_train_epochs=config_dict['hyperparameters']["epochs"],
                                        learning_rate=config_dict['hyperparameters']["learning_rate"],
                                        weight_decay=config_dict['hyperparameters']["weight_decay"],
                                        seed=config_dict['hyperparameters']["seed"],
                                        )
    else:
        training_args = Seq2SeqTrainingArguments(output_dir=config_dict['output_model_dir'],
                                        overwrite_output_dir=True,
                                        evaluation_strategy=config_dict['evaluation_strategy'],
                                        eval_steps=config_dict["eval_steps"],
                                        eval_accumulation_steps=config_dict["eval_accumulation_steps"],
                                        logging_dir=config_dict['logging_dir'],
                                        logging_strategy=config_dict['logging_strategy'],
                                        logging_steps=config_dict['logging_steps'],
                                        predict_with_generate=config_dict['predict_with_generate'],
                                        generation_max_length=config_dict['generation_max_length'],
                                        save_total_limit=config_dict['save_total_limit'],
                                        metric_for_best_model=config_dict['metric_for_best_model'],
                                        load_best_model_at_end=config_dict['load_best_model_at_end'],
                                        per_device_eval_batch_size=config_dict["eval_batch_size"],
                                        per_device_train_batch_size=config_dict['hyperparameters']["batch_size"],
                                        bf16=config_dict['hyperparameters']['bf16'],
                                        num_train_epochs=config_dict['hyperparameters']["epochs"],
                                        learning_rate=config_dict['hyperparameters']["learning_rate"],
                                        weight_decay=config_dict['hyperparameters']["weight_decay"],
                                        seed=config_dict['hyperparameters']["seed"],
                                        )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if config_dict['compute_metrics'] else None
    )
    # configure early stopping if need be
    if config_dict['use_early_stopping']:
        callback = EarlyStoppingCallback(early_stopping_patience=config_dict['early_stopping_patience'])
        trainer.add_callback(callback)

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
        # create output saving directory if needed
        if not os.path.exists(config_dict['output_model_dir']):
            os.makedirs(config_dict['output_model_dir'])
        # save model
        trainer.save_model(config_dict['output_model_dir'])
        
    if args.save_tokenizer:
        # create output saving directory if needed
        if not os.path.exists(config_dict['output_tokenizer_dir']):
            os.makedirs(config_dict['output_tokenizer_dir'])
        # save tokenizer
        trainer.tokenizer.save_pretrained(config_dict['output_tokenizer_dir'])
    
    if args.save_arguments:
        # create output saving directory if needed
        if not os.path.exists(config_dict['output_training_args_dir']):
            os.makedirs(config_dict['output_training_args_dir'])
        # save training config
        with open(os.path.join(config_dict['output_training_args_dir'], 'training_config.yaml'), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    return



def main(args):
    # LM Fine-tuning pipeline
    # --- check cuda availability
    check_cuda(args)
    # --- Load yaml using args.config
    config_dict = load_config(args)
    # --- Load raw datasets from CSVs
    raw_train_dataset, raw_validation_dataset = load_raw_train_validation_datasets(args, config_dict)
    # --- Initialize tokenizer
    tokenizer = init_tokenizer(args, config_dict)
    # tokenize datasets
    train_tokenized_dataset = tokenize_datasets(args, config_dict, tokenizer, raw_train_dataset)
    val_tokenized_dataset = tokenize_datasets(args, config_dict, tokenizer, raw_validation_dataset)
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
    trainer = init_trainer(args, config_dict, model, tokenizer, train_tokenized_dataset, val_tokenized_dataset)
    # --- Initialize wandb
    init_wandb(args, config_dict)
    # --- Set Seed
    set_seed(args, config_dict)
    # --- Train model and save
    train_and_save_model(args, config_dict, trainer)
    return


if __name__ == "__main__":
    # --- Instantiate Argument Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to configuration YAML file", required=True)

    parser.add_argument("--save_model", action='store_true', help="Option - save model after training")
    parser.add_argument("--dont_save_model", action='store_false', help="Option - save model after training")
    parser.add_argument("--save_tokenizer", action='store_true', help="Option - save tokenizer after training")
    parser.add_argument("--dont_save_tokenizer", action='store_false', help="Option - save tokenizer after training")
    parser.add_argument("--save_arguments", action='store_true', help="Option - save training arguments and config")
    parser.add_argument("--dont_save_arguments", action='store_false', help="Option - save training arguments and config")

    parser.add_argument("--use_wandb", action='store_true', help="Path to Wandb key for logging training stats", required=False)
    parser.add_argument("--dont_use_wandb", action='store_false', help="Path to Wandb key for logging training stats", required=False)
    parser.add_argument("--use_deepspeed", action='store_true', help="Option - use deepspeed during training", required=False)
    parser.add_argument("--dont_use_deepspeed", action='store_false', help="Option - don't use deepspeed during training", required=False)
    parser.set_defaults(feature=True)

    global args  # set global args scope potential for gpu diagnostics
    args = parser.parse_args()
    main(args)
