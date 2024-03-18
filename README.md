Repository containing code for training/finetuning HuggingFace transformers with wandb logging.

I created this repository to fine-tune huggingface LMs in a distributed environment with DeepSpeed (as an .ipynb file does not allow you to do such)

## EACL Paper Disclaimer ## 

If you're coming from the paper: Generation, Distillation and Evaluation of Motivational Interviewing-Style
Reflections with a Foundational Language Model (https://aclanthology.org/2024.eacl-long.75.pdf), welcome!

This code was used to train and test the distilled reflection models explained in that paper. Feel free to email me at andrew.brown.csf@gmail.com if you have any questions related to this.

## Quick Start ##
1. Create a config in the configs/ directory for the type of model you want to train (use a previous config to create yours)
2. Configure scripts/run.sh bash script to include your config and some desired flags for trainHFDS.py (I use the run.sh to set up hyperparamter sweeps so I can queue all the jobs in a background environment)
3. The script can train using either one GPU or distributed with DeepSpeed
4. If you want to use wandb, set an environment variable to WANDB_KEY with your corresponding key
5. Run run.sh with the appropriate python environment and get to training!

## Repository Explanation by Directory ##
## configs ## 
Yaml files containing the setup for model setup, hyperparameters, training, validation, wandb runs/sweeps, and inferencing.

Inside this directory there are many configs that I have used for my research at UofT. Please use them as an example for your own training.

## data ## 
Directory to store training/validation data. .gitkeep file is in here since most data we fine-tune with is private

## models ## 
Directory to store trained/untrained model weights .gitkeep file is in here since the weights are too large and do not belong in an open source environment.

## notebooks ## 
Python notebook (.ipynb) files which are used by me to inference models during testing. This code could be used for reference but is just in this repository for ease of access.

## scripts ## 
The main body of this repository. Here is a file-by-file breakdown:

### trainHFDS.py ###
A Language model fine-tuning script. With a complete pipeline
This script:

- checks cuda availability
- loads a .yaml config and dataset
- initializes tokenizer
- tokenizes datasets
- initializes a huggingface model (with optimizer, scheduler, and dataloader)
- initializes a huggingface trainer
- initialize wandb logger
- trains and saves the model weights

This is all done via a main driver function and meant to be run by using the scripts/run.sh script

### trainingUtils.py ###
helper functions which are utilized by trainHFDS.py
- config loading
- setting all seeds for training
- formatting time for logging
- cuda and GPU diagnositcs
- wandb initialization

### gptTrain.py ###
A separate training script created for fine-tuning GPT-2 
This code is similar to trainHFDS but does not use a Huggingface trainer class when fine-tuning (the training loop is written manually)

### gpt2Dataset.py ###
A class declaration written for finetuning with GPT-2 for scripts/gptTrain.py

### run.sh ###
The main script for training models
Model training jobs can be queued in here using either single or multi GPU setups
