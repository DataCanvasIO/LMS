import csv
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, TextGenerationPipeline, pipeline


def load(data_path):
    dataset = load_dataset("csv", data_files=data_path, split="train")
    return dataset


def infer(model_path, datalist):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        predict = []
        if hasattr(model, 'chat'):
            for text in tqdm(datalist):
                response, history = model.chat(tokenizer, text, history=None)
                predict.append(response)
            return predict
        else:
            pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, torch_dtype=torch.float16)
            for text in tqdm(datalist):
                out = pipe(text, max_new_tokens=128)
                predict.append(out[0]["generated_text"][len(text):])
            return predict
    except:
        pipe = pipeline("text2text-generation", model=model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        predict = []
        for text in tqdm(datalist):
            out = pipe(text, max_new_tokens=128)
            predict.append(out[0]["generated_text"][len(text):])
        return predict


def manual_eval(model_path, input_path, output_path):
    # 数据集加载s
    dataset = load(input_path)

    # 模型推理预测
    predict = infer(model_path, dataset['input'])

    # 结果写入文件
    with open(output_path, 'w', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(['input', 'Expected', 'Output'])
        for input, Expected, Output in zip(dataset["input"], dataset["Expected"], predict):
            writer.writerow([input, Expected, Output])
    print("done")
