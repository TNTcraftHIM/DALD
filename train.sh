#! /usr/bin/env bash

dataset_name="wildchat"
target_model_name="GPT-4"
scoring_model_name="llama3-8b"
output_model_dir="./ckpt"
num_samples=500


accelerate launch scripts/train_dald.py --train_dataset_name ${dataset_name} \
    --target_model_name ${target_model_name} --scoring_model_name ${scoring_model_name}\
    --output_model_dir ${output_model_dir} --num_samples ${num_samples}