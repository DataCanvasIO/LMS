from lms.runtime.prune.sparsegpt import prune


def main():
    seed = 1
    nsamples = 1
    seqlen = 1024
    sparsity = "2:4"
    model_save_path = "tmp/pruned_model"
    tokenizer_save_path = "tmp/pruned_model"

    model_path = "/Users/dev/Documents/APS/models/huggingface/opt-125m"
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
          tokenizer_save_path=tokenizer_save_path,
          device='cpu'
          )

if __name__ == "__main__":
    main()
