import sys


def prune(
        model_path,
        model_type,
        model_save_path,
        pruning_ratio,
        pruner_type,
        block_attention_layer_start,
        block_attention_layer_end,
        block_mlp_layer_start,
        block_mlp_layer_end,
        device,
        eval_before_prune=False,
        eval_after_prune=False,
        seed=42,
):
    """Prune model using LLM-Pruner algorithm.

    :param model_path: Model dir or huggingface model name. Only support llama, vicuna, bloom, baichuan models.
    :param model_save_path: Specify the directory to save the result model.
    :param pruning_ratio: Pruning ratio
    :param pruner_type: One of l2, l1, taylor.
    :param block_attention_layer_start: Start layer of block attention layers
    :param block_attention_layer_end: End layer of block attention layers
    :param block_mlp_layer_start: Start layer of block mlp layers
    :param block_mlp_layer_end: End layer of block mlp layers
    :param device: Device to use. For example: cpu, cuda, cuda:0, cuda:1, etc.
    :param eval_before_prune: Eval before prune.
    :param eval_after_prune: Eval after prune.
    :param seed: Seed.
    :return:
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pruning LLaMA (huggingface version)")

    # argument for parsing
    parser.add_argument(
        "--base_model",
        type=str,
        default="decapoda-research/llama-7b-hf",
        help="base model name",
    )
    parser.add_argument(
        "--save_ckpt_log_name",
        type=str,
        default="llama_prune",
        help="the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}",
    )
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.5, help="pruning ratio"
    )
    parser.add_argument("--pruner_type", type=str, default="l2", help="pruner type")

    # argument for generation
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="top p")
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="max sequence length"
    )

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument("--channel_wise", action="store_true", help="channel wise")
    parser.add_argument("--block_wise", action="store_true", help="block wise")
    parser.add_argument("--layer_wise", action="store_true", help="layer wise")
    parser.add_argument(
        "--layer", type=int, default=12, help="remain the previous n layers"
    )

    parser.add_argument(
        "--block_attention_layer_start",
        type=int,
        help="start layer of block attention layers",
        default=3,
    )
    parser.add_argument(
        "--block_attention_layer_end",
        type=int,
        help="end layer of block attention layers",
        default=31,
    )
    parser.add_argument(
        "--block_mlp_layer_start",
        type=int,
        help="start layer of block mlp layers",
        default=3,
    )
    parser.add_argument(
        "--block_mlp_layer_end",
        type=int,
        help="end layer of block mlp layers",
        default=31,
    )

    parser.add_argument(
        "--iterative_steps",
        type=int,
        default=1,
        help="Iteration step for pruning. Default=1",
    )
    parser.add_argument(
        "--grouping_strategy",
        type=str,
        default="sum",
        help="Reduce method for grouping",
    )
    parser.add_argument(
        "--global_pruning", action="store_true", help="whether global pruning"
    )
    parser.add_argument(
        "--taylor",
        type=str,
        default="param_first",
        help="choose from [vectorize, param_second, param_first, param_mix]",
    )
    parser.add_argument("--num_examples", type=int, default=10)

    # general argument
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--test_before_train", action="store_true", help="whether test before train"
    )
    parser.add_argument("--eval_device", type=str, default="cuda", help="eval device")
    parser.add_argument(
        "--test_after_train", action="store_true", help="whether test after train"
    )

    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--save_model", action="store_true", help="if save model")
    args = parser.parse_known_args()[0]

    import torch

    torch_version = float(".".join(torch.__version__.split(".")[:2]))

    # class Args(dict):
    #     def __getattr__(self, name):
    #         return self[name]

    #     def __setattr__(self, name, value):
    #         self[name] = value

    # args = Args()

    args.torch_version = torch_version

    args.base_model = model_path
    args.save_ckpt_log_name = model_save_path

    try:
        print(f"pruning_ratio is {pruning_ratio}")
        assert isinstance(pruning_ratio, float) and 0 < pruning_ratio < 1
    except:
        print(
            f"pruning_ratio should be a float between 0 and 1, exclusive.\nNow it's {pruning_ratio}")
        sys.exit(1)

    args.pruning_ratio = pruning_ratio
    args.pruner_type = pruner_type

    args.block_wise = True

    args.block_attention_layer_start = block_attention_layer_start
    args.block_attention_layer_end = block_attention_layer_end
    args.block_mlp_layer_start = block_mlp_layer_start
    args.block_mlp_layer_end = block_mlp_layer_end

    args.device = device
    args.test_before_train = eval_before_prune
    args.eval_device = device
    args.test_after_train = eval_after_prune
    if seed is None:
        seed = 42
    args.seed = seed
    args.save_model = True

    from pathlib import Path

    model_save_path = Path(model_save_path)
    model_save_path.mkdir(exist_ok=True, parents=True)

    print(f"============== The model to prune is: {model_path}")

    if model_type == "llama":
        from .hf_prune import main
    elif model_type == "vicuna":
        from .hf_prune import main
    elif model_type == "bloom":
        from .bloom import main
    elif model_type == "baichuan":
        from .baichuan import main
    else:
        print("Please specify the right model type in model path.")
        sys.exit()

    # if "llama" in model_path.lower():
    #     from .hf_prune import main
    # elif "vicuna" in model_path.lower():
    #     from .hf_prune import main
    # elif "bloom" in model_path.lower():
    #     from .bloom import main
    # elif "baichuan" in model_path.lower():
    #     from .baichuan import main
    # else:
    #     print("Please specify the model name in model path.")
    #     sys.exit()

    main(args)

    print("All the work has been done.")
