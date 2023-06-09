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

# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp1.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp2.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb

# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp3.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp4.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp5.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp6.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp7.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp8.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb

# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp1.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp2.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp3.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp4.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp5.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp6.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp7.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp8.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb

# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp1.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp2.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp3.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp4.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp5.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp6.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp7.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp8.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb

# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp8.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp1.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp2.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp3.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp4.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp5.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp6.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp7.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb

# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium-echo.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large-echo.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
# deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl-echo.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb

deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-small/gpt2-small-exp5-complex.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-medium/gpt2-medium-exp7-complex.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-large/gpt2-large-exp7-complex.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb
deepspeed scripts/gptTrainHFDS.py -c /home/ubuntu/transformer-fine-tune/configs/gpt2-xl/gpt2-xl-exp7-complex.yaml --save_model --save_tokenizer --dont_save_arguments --use_deepspeed --use_wandb