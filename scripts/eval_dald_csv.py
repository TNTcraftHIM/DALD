import argparse

import pandas as pd
import torch
import random
import tqdm
import json
import numpy as np

from peft import PeftModel
from sklearn.metrics import roc_auc_score, roc_curve

from model import load_model, load_tokenizer
from fast_detect_gpt import get_sampling_discrepancy, get_sampling_discrepancy_analytic
from metrics import get_roc_metrics, get_precision_recall_metrics
from data_builder import load_data

def process_p_values_and_labels_odd(answer_labels, results_list):
    # 计算 AUROC
    auroc = roc_auc_score(answer_labels, results_list)
    fpr, tpr, thresholds = roc_curve(answer_labels, results_list)
    accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}
    print("auroc: {:.4f}; ".format(auroc) + "; ".join(
        ["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()]))

    return auroc


def process_p_values_and_labels(answer_labels, results_list):
    # 初始化 AUROC 列表
    auroc_list = []

    # 确保标签和结果列表长度匹配
    assert len(answer_labels) == len(results_list)

    # 用于记录已配对的索引
    used_indices = set()

    # 处理每对标签
    for i in range(len(answer_labels)):
        if i in used_indices:
            continue

        # 当前标签
        current_label = answer_labels[i]

        # 找到下一个未使用的相反标签的索引
        for j in range(len(answer_labels)):
            if answer_labels[j] != current_label and j not in used_indices:
                # 获取当前对的 p 值
                p_values_0 = [results_list[i] if current_label == 0 else results_list[j]]
                p_values_1 = [results_list[j] if current_label == 0 else results_list[i]]

                # 合并 p 值和标签
                combined_p_values = p_values_0 + p_values_1
                combined_labels = [0] * len(p_values_0) + [1] * len(p_values_1)

                # 计算 AUROC
                auroc = round(roc_auc_score(combined_labels, combined_p_values), 6)
                fpr, tpr, thresholds = roc_curve(combined_labels, combined_p_values)
                accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}
                auroc_list.append(auroc)
                # print("auroc: {:.4f}; ".format(auroc) + "; ".join(
                #     ["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()]))

                # 标记已使用的索引
                used_indices.update([i, j])
                break

    return auroc_list

def eval_fastdetect(args):
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.eval_dataset_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    if args.weight_path is not None:
        scoring_model = PeftModel.from_pretrained(scoring_model, args.weight_path)

    scoring_model.eval()
    reference_model_name = args.reference_model_name
    device = args.device
    cache_dir = args.cache_dir
    # dataset = "WildChat"
    
    reference_tokenizer = load_tokenizer(reference_model_name, args.eval_dataset_name, cache_dir)
    reference_model = load_model(reference_model_name, device, cache_dir)
    reference_model.eval()
    
    
    # load data
    print('Reading csv from data/test_data.csv...')
    csv = pd.read_csv('data/test_data.csv', encoding='utf-8')
    answer_labels = [1 if 'MGT' in label else 0 for label in csv['label']]
    data = {"original": csv[csv['label'] == 'HWT']['text'].tolist(), "sampled": csv[csv['label'] == 'MGT']['text'].tolist()}
    n_samples = len(data["sampled"])
    # evaluate criterion
    if args.discrepancy_analytic:
        name = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        # tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding="max_length", max_length=200, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            
            tokenized = reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
            # tokenized = reference_tokenizer(original_text, return_tensors="pt", padding="max_length", max_length=200, return_token_type_ids=False).to(args.device)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = reference_model(**tokenized).logits[:, :-1]
            original_crit = criterion_fn(logits_ref, logits_score, labels)
        # sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        # tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding="max_length", max_length=200, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
            # tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding="max_length", max_length=200, return_token_type_ids=False).to(args.device)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = reference_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    predictions_temp = predictions.copy()
    # process predictions to match answer_labels
    predictions_combined = []
    answer_labels_temp = answer_labels.copy()
    for answer in answer_labels_temp:
        if answer == 0:
            try:
                predictions_combined.append(predictions_temp['real'].pop(0))
            except IndexError:
                # if there are no more real predictions, remove it from answer_labels
                answer_labels.remove(answer)
        else:
            try:
                predictions_combined.append(predictions_temp['samples'].pop(0))
            except IndexError:
                # if there are no more sample predictions, remove it from answer_labels
                answer_labels.remove(answer)
    # print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    # fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    # p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    auroc_list = process_p_values_and_labels(answer_labels, predictions_combined)
    all_auroc = process_p_values_and_labels_odd(answer_labels, predictions_combined)
    print(f"Criterion {name}:")
    print("avg_auroc: {:.4f}".format(sum(auroc_list) / len(auroc_list))
          + "; ".join(["std_auroc: {:.4f}".format(np.std(auroc_list))]))
    print("all_auroc: ", all_auroc)
    # results

    dataset_file = args.eval_dataset_file.split("/")[-1]
    output_file = f"{args.output_path}/{dataset_file}"
    results_file = f'{output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'all_auroc': all_auroc, 'avg_auroc': sum(auroc_list) / len(auroc_list), 'std_auroc': np.std(auroc_list), 'auroc_list': auroc_list}}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="llama2-7b")
    parser.add_argument('--scoring_model_name', type=str, default="llama2-7b")
    parser.add_argument('--weight_path', type=str, default="./ckpt/checkpoint-1860")
    parser.add_argument('--target_model_name', type=str, default="ChatGPT")
    parser.add_argument('--output_path', type=str, default="./")
    
    parser.add_argument('--eval_dataset_name', type=str, default="xsum")
    parser.add_argument('--eval_dataset_file', type=str, default="/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt3.5turbo0301/data/xsum_gpt-3.5-turbo-0301")

    parser.add_argument('--discrepancy_analytic', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    eval_fastdetect(args)