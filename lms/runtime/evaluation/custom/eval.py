import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline


def load_class_object(script_file):
    """
    load a valid custom class object from script file
    :param script_file:
    :return:
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("lms.runtime.evaluation.custom.dynamic", script_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lms.runtime.evaluation.custom.dynamic"] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise Exception("There is no file in the path:" + script_file)
    class_object = None
    for name in dir(module):
        attr = getattr(module, name)
        if isinstance(attr, type):
            if hasattr(attr, 'load') and hasattr(attr, 'eval'):
                class_object = attr
            else:
                continue
        else:
            continue

    if class_object is None:
        raise Exception("The valid class object of custom evaluation couldn't be found at file:" + script_file)

    return class_object


def infer(model_path, datalist):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, torch_dtype=torch.float16)
    except Exception:
        pipe = pipeline("text2text-generation", model=model_path, device_map="auto", trust_remote_code=True,
                        torch_dtype=torch.float16)
    predict = []
    for text in tqdm(datalist):
        out = pipe(text, max_new_tokens=64)
        predict.append(out[0]["generated_text"][len(text):])
    return predict


def do_eval(script_file, data_path, model_path, output_path, accuracy_script_path):
    custom_class_object = load_class_object(script_file)
    # 数据集加载
    dataset = custom_class_object.load(data_path)
    # 模型推理预测
    predict = infer(model_path, dataset["question"])
    # 数据集评估
    result = custom_class_object.eval(predict, dataset["answer"], accuracy_script_path)
    results_metric = {"model": model_path, "benchmarks": [{"benchmark_name": script_file, "metrics": result}]}
    with open(output_path, 'w') as write_f:
        write_f.write(json.dumps(results_metric, indent=4, ensure_ascii=False))
    print(result)
    return results_metric
