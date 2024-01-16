#!/bin/bash

# Here are some example commands to run these scripts

# Using python
# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl-echo.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl-beta.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb

# Using deepspeed 

#deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp5-complex.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp7-complex.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp7-complex.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp7-complex.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb