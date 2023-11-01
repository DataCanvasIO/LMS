import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig

from lms.runtime.prune.llm_pruner import load_model


def load_class_object(script_file):
    """
    load a valid custom class object from script file
    :param script_file:
    :return:
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "lms.runtime.deploy.custom.dynamic", script_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lms.runtime.deploy.custom.dynamic"] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise Exception("There is no file in the path:" + script_file)
    class_object = None
    for name in dir(module):
        attr = getattr(module, name)
        if isinstance(attr, type):
            if hasattr(attr, "load"):
                class_object = attr
            else:
                continue
        else:
            continue

    if class_object is None:
        raise Exception(
            "The valid class object of custom evaluation couldn't be found at file:"
            + script_file
        )

    return class_object


def load_pipeline_for_deepspeed(model_path, tokenizer_kwargs={}, model_kwargs={}, learning_type="TEXT2TEXT-GENERATION"):

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True, **tokenizer_kwargs
    )
    if config.model_type == "t5":
        from transformers import T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **model_kwargs
        )
        # model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, **model_kwargs)

    pipeline_kwargs = dict()
    # rectify task
    if config.architectures:
        for architecture in config.architectures:
            if architecture.endswith("ForCausalLM"):
                learning_type = "TEXT-GENERATION"
                break
            elif architecture.endswith("ForConditionalGeneration"):
                learning_type = "TEXT2TEXT-GENERATION"
                break

    pipe = pipeline(
        task=learning_type.lower(),
        # config=config,
        tokenizer=tokenizer,
        model=model,
        trust_remote_code=True,
        model_kwargs={"ignore_mismatched_sizes": True},
        **pipeline_kwargs,
    )

    return pipe


def load_pipeline(
    model_path, tokenizer_kwargs={}, model_kwargs={}, learning_type="TEXT2TEXT-GENERATION", dtype=None, device=None
):
    """
    load a pipeline model
    :param model_path:
    :param learning_type:
    :param dtype:
    :param device:
    :return:
    """
    if not os.path.exists(model_path):
        raise Exception("This path must be exists" % model_path)
    model, tokenizer = load_model(model_path, device)
    if model is not None:
        pipeline_kwargs = dict()
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map is None:
            pipeline_kwargs["device"] = device

        pipe = pipeline(
            task=learning_type.lower(),
            tokenizer=tokenizer,
            model=model,
            trust_remote_code=True,
            model_kwargs={"ignore_mismatched_sizes": True},
        )

        return pipe
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        quantized = False
        if hasattr(config, "quantization_config"):
            if not config.quantization_config.get("disable_exllama", None):
                # if device == 'cpu':
                #     raise Exception('quantized model must specify gpu')
                config.quantization_config["disable_exllama"] = True
                quantized = True
        else:
            if dtype == "int4":
                model_kwargs["load_in_4bit"] = True
            elif dtype == "int8":
                model_kwargs["load_in_8bit"] = True
            elif dtype == "fp16":
                model_kwargs["torch_dtype"] = torch.float16

        # rectify task
        if config.architectures:
            for architecture in config.architectures:
                if architecture.endswith("ForCausalLM"):
                    learning_type = "TEXT-GENERATION"
                    break
                elif architecture.endswith("ForConditionalGeneration"):
                    learning_type = "TEXT2TEXT-GENERATION"
                    break

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True, **tokenizer_kwargs
        )
        if config.model_type == "t5":
            from transformers import T5ForConditionalGeneration

            if quantized:
                model = T5ForConditionalGeneration.from_pretrained(
                    config, **model_kwargs
                )
            else:
                model = T5ForConditionalGeneration.from_pretrained(
                    model_path, trust_remote_code=True, **model_kwargs
                )
        else:
            if quantized:
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True, **model_kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True, **model_kwargs
                )

        pipeline_kwargs = dict()
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map is None:
            pipeline_kwargs["device"] = device

        pipe = pipeline(
            task=learning_type.lower(),
            config=config,
            tokenizer=tokenizer,
            model=model,
            trust_remote_code=True,
            model_kwargs={"ignore_mismatched_sizes": True},
            **pipeline_kwargs,
        )

        return pipe


def read_infer_config(infer_config_path):
    if infer_config_path is not None:
        try:
            with open(infer_config_path, encoding="utf-8") as f:
                import json as jzon

                infer_config = jzon.load(f)
                # validate prompt_role
                if "prompt_role" in infer_config:
                    if (
                        "User" not in infer_config["prompt_role"]
                        or "Assistant" not in infer_config["prompt_role"]
                    ):
                        raise Exception(
                            "The prompt_role must contains 'User','Assistant'."
                        )
                return infer_config
        except FileNotFoundError as e:
            raise Exception("The path of infer config isn't exist") from e
        except (OSError, ValueError, KeyError) as e:
            raise Exception("failed to parse infer config file") from e
    else:
        return {}


def load_config(model_path):
    config = None
    config_path = "%s/%s" % (model_path, "config.json")
    if os.path.isfile(config_path):
        import json

        with open(config_path) as f:
            config = json.load(f)
    return config


if __name__ == "__main__":
    # dd = load_pipeline('/Users/dev/Documents/APS/models/huggingface/opt-125m')
    dd = load_pipeline("/Users/dev/Documents/APS/models/huggingface/t5-v1_1-small")
    print(dd("hi"))
