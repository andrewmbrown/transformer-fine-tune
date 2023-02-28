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


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


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


def init_tokenizer(args, config_dict):
    """
    DESC:   Initialize GPT2 tokenizer (byte-level BPE tokenizer)
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
    OUTPUT: tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
    """
    assert args.path_to_tokenizer is not None, "Please provide a path to tokenizer"
    # model name is the same as tokenizer, so we can use it to load the tokenizer
    # bos_token, eos_token, and pad_token are custom and added to the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config_dict['model_name'], bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    return tokenizer


def load_dataset(args, config_dict, tokenizer, df, mode="Random"):
    """
    DESC:   Load dataframe into GPT2Dataset objects (tokenized with len and getitem overrides)
            Then load into torch DataLoader objects (with sampler type and batch_size)
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
            df (pd.DataFrame) dataframe containing any data
    OUTPUT: dataloader (GPT2Dataset) object containing any data
            val_dataset (GPT2Dataset) dataset containing validation data
    """
    assert config_dict['max_length'] is not None, "Please provide a max length for the dataset"
    assert tokenizer is not None, "Please provide a tokenizer" 
    # Create custom dataset objects
    dataset = GPT2Dataset(df, tokenizer, config_dict['max_length'])
    # Load datasets into torch DataLoader objects
    # take training samples in random order, for validation, sequential order (no shuffling needed))
    if mode == "Random":
        dataloader = DataLoader(dataset,
                            sampler=RandomSampler(dataset),
                            batch_size=config_dict['batch_size'])
    elif mode == "Sequential":
        dataloader = DataLoader(dataset,
                            sampler=SequentialSampler(dataset),
                            batch_size=config_dict['batch_size'])
    else:
        assert False, "Please provide a valid mode to create a DataLoader"
    return dataloader


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


def train_and_validate(args, config_dict, model, tokenizer, train_dataloader, val_dataloader, optimizer, scheduler):
    """
    DESC:   Train and validate huggingface LM
    INPUT:  args (argparse.ArgumentParser)
            config_dict (dict) dictionary containing training configs
            model (GPT2LMHeadModel) untrained GPT2 model
            tokenizer (GPT2Tokenizer) tokenizer for GPT2 model
            train_dataloader (torch.DataLoader) dataloader containing training data
            val_dataloader (torch.DataLoader) dataloader containing validation data
            optimizer (torch.optim.xxx or transformers.xxx) optimizer
            scheduler (get_linear_schedule_with_warmup) scheduler
    OUTPUT: model (GPT2LMHeadModel) trained GPT2 model
    """
    # start global timer
    total_t0 = time.time()

    training_stats = []

    # unpack some args and config params
    device = args.device
    epochs = config_dict['num_epochs']
    sample_every = config_dict['sample_every']

    model = model.to(device)

    for epoch_i in range(0, epochs):
        """
        Training Loop
        """
        wandb.watch(model)
        print(f"---Epoch {epoch_i + 1} of {epochs}---")
        print("---Training...---")
        
        # start epoch timer
        t0 = time.time()
        
        total_train_loss = 0

        # sets model into train mode, not actual backprop
        # dropout and batchnorm behave differently
        # opposite of model.eval() for inference mode
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            # grab input tokens, labels, and masks
            input_tokens = batch[0].to(device)
            # in this case, we're generating text,
            # so label tokens are the input tokens shifted
            label_tokens = batch[0].to(device)
            attn_masks = batch[1].to(device)
            
            # clear any gradients from model tensors
            # prevents any gradient accumulation
            model.zero_grad()
            
            # forward pass
            outputs = model(input_tokens,
                            labels=label_tokens,
                            attention_mask=attn_masks,
                            token_type_ids=None
                        )
            
            # grab loss from outputs
            loss = outputs[0]
            
            batch_loss = loss.item()  # detach from device with item
            total_train_loss += batch_loss
            
            # get sample every x batches
            if step % sample_every == 0 and not step == 0:
                # calculate elapsed time and print statistics
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))
                
                # set model to inference mode for testing
                model.eval()
                
                # sample model with generate() using no tokens, just let it generate
                sample_outputs = model.generate(bos_token_id=random.randint(1,30000),
                                                do_sample=True,
                                                top_k=50,
                                                max_length=200,
                                                top_p=0.95,
                                                num_return_sequences=1
                                            )
                
                for i, sample_output in enumerate(sample_outputs):
                    out = tokenizer.decode(sample_output, skip_special_tokens=True)
                    print(f"{i}: {out}")
                
                # back to train mode
            
            # backpropagation step
            # computes dloss/dx for every parameter x which has requires_grad=True.
            # updates gradient values
            # x.grad += dloss/dx
            loss.backward()
            
            # step optimizer
            # updates the value of x using the gradient x.grad
            # x += -lr * x.grad
            optimizer.step()
            
            # step scheduler
            # tells scheduler to increase learning rate
            # using our warmup steps
            scheduler.step()
            
        print("---Done Training Epoch!---")
        # calculate average loss over all batches
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # measure how long the epoch took
        training_time = format_time(time.time() - t0)
        
        print(f"---Average training loss {avg_train_loss} ---")
        print(f"---Training epoch took {training_time} ---")
        
        """
        Validation
        """
        print("---Running Validation...---")
        
        # start batch timer
        t0 = time.time()
        
        # set model to inference mode
        model.eval()
        
        total_eval_loss = 0
        nb_eval_steps = 0
        
        # evaluate data for one epoch
        for batch in val_dataloader:
            # grab input tokens, labels, and masks
            input_tokens = batch[0].to(device)
            # in this case, we're generating text,
            # so label tokens are the input tokens shifted
            label_tokens = batch[0].to(device)
            attn_masks = batch[1].to(device)
            
            # freeze gradients
            with torch.no_grad():
                outputs = model(input_tokens,
                                attention_mask=attn_masks,
                                labels=label_tokens)
                
                loss = outputs[0]
            
            batch_loss = loss.item()
            total_eval_loss += batch_loss
            
        avg_val_loss = total_eval_loss / len(val_dataloader)
        validation_time = format_time(time.time() - t0)
        
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Validation took: {validation_time}")
        
        # save all training statistics from the epoch
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Validation Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
        # log training data to wandb as well
        wandb.log({
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Validation Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
        

    print("---Training Complete!")
    print(f"---Total training time took {format_time(time.time()-total_t0)}")
    return model


def main():
    # --- Instantiate Argument Parser ---
    # path_to_data, 
    parser = argparse.ArgumentParser()
    global args
    args = argparse.parse_args()



if __name__ == "__main__":
    main()