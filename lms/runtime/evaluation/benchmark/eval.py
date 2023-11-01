import os
import torch
from lms.runtime.evaluation.benchmark.eval_dataset import ARCDataset, MMLUDataset, CMMLUDataset, CEvalDataset, \
    AGIEvalDataset, \
    BBHDataset
from lms.runtime.evaluation.benchmark.eval_metric import AccEvaluator, MCAccEvaluator
import argparse
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Run an evaluation task')
    parser.add_argument('--model_path', help='model_path')
    parser.add_argument('--task', help='task')
    parser.add_argument('--output_path', help='output_path')
    args = parser.parse_args()
    return args


def trunk(text, text_length=800):
    return str(text[len(text) - text_length:])


def infer(model_path, datalist,task):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, torch_dtype=torch.float16)
    except:
        pipe = pipeline("text2text-generation", model=model_path, device_map="auto",trust_remote_code=True, torch_dtype=torch.float16)
    predict = []
    datalist = list(map(trunk, datalist))
    for text in tqdm(datalist):
        if task=="BigBench":
            out = pipe(text, max_new_tokens=32)
        else:
            out = pipe(text, max_new_tokens=4)
        predict.append(out[0]["generated_text"][len(text):])
    return predict


task_map = {"ARC": ARCDataset, "MMLU": MMLUDataset, "CMMLU": CMMLUDataset, "ceval": CEvalDataset,
            "AGIEval": AGIEvalDataset, "BigBench": BBHDataset}
eval_map = {"ARC": AccEvaluator, "MMLU": AccEvaluator, "CMMLU": AccEvaluator, "ceval": AccEvaluator,
            "AGIEval": AccEvaluator, "BigBench": MCAccEvaluator}


def do_eval(task, model_path, output_path):
    tasks = task.split(",")

    results_metric = {"model": model_path, "benchmarks": []}
    # 对所有benchmark任务进行评估
    for task in tasks:
        # 数据集加载
        print(task)
        dataset = task_map[task].load()

        # 模型推理预测
        predict = infer(model_path, dataset["question"],task)

        # 数据集评估
        Acc = eval_map[task](os.path.dirname(__file__) + "/accuracy.py")
        result = Acc.eval(predict, dataset["answer"])
        results_metric["benchmarks"].append({"benchmark_name": task, "metrics": result})

    # 评估结果写入文件
    with open(output_path, 'w') as write_f:
        write_f.write(json.dumps(results_metric, indent=4, ensure_ascii=False))
    return results_metric
