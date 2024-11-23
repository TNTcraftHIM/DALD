#! /usr/bin/env bash

target_model_name="GPT-4"
reference_model_name="llama3-8b"
scoring_model_name="llama3-8b"
weight_path="./ckpt/checkpoint-125"
eval_dataset_name="Pubmed"
eval_dataset_file="./exp_gpt4-0613/data/pubmed_gpt-4-0613"

python scripts/eval_dald.py --reference_model_name ${reference_model_name} --scoring_model_name ${scoring_model_name} \
    --weight_path ${weight_path} --target_model_name ${target_model_name} --eval_dataset_name ${eval_dataset_name} \
    --eval_dataset_file ${eval_dataset_file} --discrepancy_analytic 