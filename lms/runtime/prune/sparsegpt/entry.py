from lms.runtime.prune.sparsegpt import util


def prune(model_path, tokenizer_path=None, only_check_model=False,
          layer_name_start=None, layer_name_stop=None, seed=1, nsamples=64,
          seqlen=1024, sparsity="2:4", model_save_path=None, tokenizer_save_path=None, device=None):
    """
    Prune model using sparsegpt algorithm.

    :param model_path: Model dir or huggingface model name.
    :param tokenizer_path: Tokenizer dir or huggingface model name. If None, will use `model_path`.
    :param only_check_model: If True, only print the model structure without running the prune algorithm.
    :param layer_name_start: Specify the layer name from which to start pruning.
    :param layer_name_stop: Specify the layer name util which to stop pruning. Will include this layer.
    :param seed: The random seed to select calibration samples.
    :param nsamples: The number of samples used for pruning.
    :param seqlen: The sequence length of each sample tokens.
    :param sparsity: The sparse ratio or the n:m sparsity paradigm.
     For example: 0.6 to specify the sparse ratio of each layer,
                  2:4 to specify 2 out of every 4 elements are zero.
    :param model_save_path: Specify the directory to save the result model.
    :param tokenizer_save_path: Specify the directory to save the tokenizer. If None, will use `model_save_path`.
    :Param device: Specify the device. For example, cpu, cuda, cuda:0, cuda:1
    :return:
    """
    import torch
    from pathlib import Path
    import sys
    # Check params

    if tokenizer_path is None:
        tokenizer_path = model_path
    if tokenizer_save_path is None:
        tokenizer_save_path = model_save_path
    prunen = 0
    prunem = 0
    try:
        if ":" in sparsity:
            prunen, prunem = [int(v.strip()) for v in sparsity.split(":")]
            assert isinstance(prunen, int) and isinstance(prunem, int)
            assert 0 < prunen < prunem
        else:
            sparsity = float(sparsity.strip())
            assert 0 < sparsity < 1
    except:
        print(
            f"Sparsity should either be a float in the range (0, 1), or the n:m form (2:4, 2:6, ..., etc.).\nNow it's {sparsity}")
        sys.exit(1)

    try:
        assert isinstance(nsamples, int) and nsamples > 1
    except:
        print(
            f"nsample should be more than one.\nNow it's {nsamples}")
        sys.exit(1)

    try:
        assert isinstance(seqlen, int) and seqlen >= 10
    except:
        print(
            f"seqlen should not be less than 10.\nNow it's {seqlen}")
        sys.exit(1)

    # Load model
    import transformers as tm

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "model_type") and config.model_type == "baichuan":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model = tm.AutoModel.from_pretrained(model_path)
    old_use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    node = util.Node(name="", module=model, verbose=True)
    model_structure = f"{'Model Stucture':=^100}\n{repr(node)}\n"
    print(model_structure)
    name2module = {k: v for k, v in model.named_modules()}
    names = [v[0] for v in model.named_modules()]
    while layer_name_start not in name2module:
        layer_name_start = input("Need to specify which layer to start to prune: ").strip()
    print(f"Set start layer to: {layer_name_start}")
    start_index = names.index(layer_name_start)

    while layer_name_stop not in names or (layer_name_stop in names and names.index(layer_name_stop) <= start_index):
        layer_name_stop = input("Need to specify which layer to stop to prune: ").strip()
    print(f"Set stop  layer to: {layer_name_stop}")
    if only_check_model:
        util.model_info(model)
        import sys

        sys.exit()

    # Load tokenizer
    tokenizer = util.load_tokenizer(tokenizer_path)

    # Load dataset
    train_batch = util.get_c4_batch(tokenizer=tokenizer, nsamples=nsamples, seed=seed, seqlen=seqlen)
    # Select device
    if str(device).startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            print("Cuda is not available, use cpu instead!")
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Auto load to device
    device_hook = util.DeviceHook(device=device)
    device_hook.attach_to_module(module=model)
    # Find modules to prune
    finder = util.ModuleFinder(start=layer_name_start, stop=layer_name_stop)
    finder.attach_to_module(module=model)
    try:
        with torch.no_grad():
            model(train_batch[:1])
    except finder.FinderStopped:
        pass
    target_modules = finder.get_matched_modules()
    # Sparse it
    util.AddSparseHook.attach_to_modules(target_modules, sparsity=sparsity, prunen=prunen, prunem=prunem)
    with torch.no_grad():
        model(train_batch)

    # Restore something
    model.config.use_cache = old_use_cache
    device_hook.remove_hooks()

    # Print model info
    util.model_info(model=model)
    if tokenizer_save_path is not None:
        tokenizer_save_path = Path(tokenizer_save_path)
        tokenizer.save_pretrained(save_directory=tokenizer_save_path)
        print(f"Tokenizer saved to: {tokenizer_save_path.absolute()}")
    else:
        print("Skip tokenizer saving. No save directory specified.")
    if model_save_path is not None:
        model_save_path = Path(model_save_path)
        model.save_pretrained(save_directory=model_save_path)
        print(f"Model saved to: {model_save_path.absolute()}")
    else:
        print("Skip model saving. No save directory specified.")
    print("All the work has been done.")


def main():
    seed = 1
    nsamples = 1
    seqlen = 1024
    sparsity = "2:4"
    model_save_path = "tmp/pruned_model"
    tokenizer_save_path = "tmp/pruned_model"

    model_path = "facebook/opt-125m"
    tokenizer_path = None
    only_check_model = False
    layer_name_start = "decoder.layers.1"
    layer_name_stop = "decoder.layers.11.fc1"

    # model_path = "decapoda-research/llama-7b-hf"
    # only_check_model = False
    # layer_name_start = "layers.1"
    # layer_name_end = "layers.29"
    prune(model_path=model_path,
          tokenizer_path=tokenizer_path,
          only_check_model=only_check_model,
          layer_name_start=layer_name_start,
          layer_name_stop=layer_name_stop,
          seed=seed,
          nsamples=nsamples,
          seqlen=seqlen,
          sparsity=sparsity,
          model_save_path=model_save_path,
          tokenizer_save_path=tokenizer_save_path)


if __name__ == "__main__":
    main()
