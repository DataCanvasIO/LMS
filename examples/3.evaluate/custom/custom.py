import argparse
import csv
import json
from typing import List
import torch
import evaluate
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline


class CustomDataset():
    @staticmethod
    def load(data_path: str):
        pre_prompt = "以下是单项选择题，请直接给出正确答案的选项。\n题目："
        post_prompt: str = "答案是："
        raw_data = []
        with open(data_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            _ = next(reader)  # skip the header
            for row in reader:
                assert len(row) == 6
                question = row[0]
                A = row[1]
                B = row[2]
                C = row[3]
                D = row[4]
                raw_data.append({
                    'question': pre_prompt + f"{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}" + post_prompt,
                    'answer': row[5],
                })
        dataset = Dataset.from_list(raw_data)
        return dataset

    #   dataset为Dataset结构，形式为：
    #   Dataset({
    #             features: ['question', 'answer'],
    #             num_rows: xxx
    #          })

    @staticmethod
    def eval(predictions: List, references: List, accuracy_script_path) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: calculated scores.
        """
        import os
        metric = evaluate.load(accuracy_script_path)

        def first_capital_postprocess(text):
            for t in text:
                if t.isupper():
                    return t
            return ''

        def preprocess(predictions, references):
            mapping_to_int_dict = {
                label: idx
                for idx, label in enumerate(set(map(str, references)))
            }
            pred_set = set(predictions)
            for pred in pred_set:
                if str(pred) not in mapping_to_int_dict.keys():
                    mapping_to_int_dict[str(pred)] = len(mapping_to_int_dict)
            golds = [mapping_to_int_dict[str(gold)] for gold in references]
            preds = [mapping_to_int_dict[str(pred)] for pred in predictions]
            return {
                'predictions': preds,
                'references': golds,
            }

        predictions = list(map(first_capital_postprocess, predictions))
        scores = metric.compute(**preprocess(predictions, references))
        result = {}
        result["acc"] = round(scores['accuracy'], 2)
        return result
        #   reuslt结构为{"xxx":0.8}  xxx如：acc,f1,bleu,rouge1等样式的自定义评估指标


def parse_args():
    parser = argparse.ArgumentParser(description='Run an evaluation task')
    parser.add_argument('--model_path', help='model_name_or_path')
    parser.add_argument('--data_path', help='data_path')
    parser.add_argument('--output_path', help='output_path')
    args = parser.parse_args()
    return args


def infer(model_path, datalist):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, torch_dtype=torch.float16)
    except:
        pipe = pipeline("text2text-generation", model=model_path, device_map="auto", trust_remote_code=True,
                        torch_dtype=torch.float16)
    predict = []
    for text in tqdm(datalist):
        out = pipe(text, max_new_tokens=64)
        predict.append(out[0]["generated_text"][len(text):])
    return predict


def do_eval(data_path, model_path, output_path, accuracy_script_path):
    # 数据集加载
    dataset = CustomDataset.load(data_path)
    # 模型推理预测
    predict = infer(model_path, dataset["question"])

    # 数据集评估
    result = CustomDataset.eval(predict, dataset["answer"], accuracy_script_path)
    results_metric = {"model": model_path, "benchmarks": [{"benchmark_name": "custom1", "metrics": result}]}
    with open(output_path, 'w') as write_f:
        write_f.write(json.dumps(results_metric, indent=4, ensure_ascii=False))
    print(result)
    return results_metric
