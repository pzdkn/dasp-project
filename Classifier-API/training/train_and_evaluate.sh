#!/usr/bin/env bash
source ../.venv/bin/activate

data_dir='data/SentimentAnalysis'
model_name_or_path='distilbert-base-uncased'
task_name = 'argument_mining'
language='en'
output_dir='../api/models/sa/distilbert'
save_steps=1000

python3 train_and_evaluate.py --do_train --do_eval --overwrite_output_dir \
        --data_dir=$data_dir --language=$language --output_dir=$output_dir \
        --save_steps=$save_steps --model_name_or_path=$model_name_or_path \
        --task_name=$task_name
