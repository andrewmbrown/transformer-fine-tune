#!/bin/bash

# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small-alpha.yaml --save_model --save_tokenizer --dont_save_arguments  --use_wandb
# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small-beta.yaml --save_model --save_tokenizer --dont_save_arguments  --use_wandb
# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small-charlie.yaml --save_model --save_tokenizer --dont_save_arguments --use_wandb
#python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small-delta.yaml --save_model --save_tokenizer --dont_save_arguments --use_wandb
# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small-echo.yaml --save_model --save_tokenizer --dont_save_arguments --use_wandb
# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium-echo.yaml --save_model --save_tokenizer --dont_save_arguments --use_wandb
# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large-echo.yaml --save_model --save_tokenizer --dont_save_arguments --use_wandb
# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl-echo.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# python ./scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl-beta.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl-echo.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb