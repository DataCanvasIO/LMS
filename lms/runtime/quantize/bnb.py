from lms.runtime.quantize import util


def bnb_quantize(model_path, tokenizer_path=None, model_save_path=None, tokenizer_save_path=None, device=None, verbose=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, pipeline
    from pathlib import Path


    if tokenizer_path is None:
        tokenizer_path = model_path
    if tokenizer_save_path is None:
        tokenizer_save_path = model_save_path
    
    if device is None:
        device = "auto"
    print(f"Using device: {device}")

    #加载模型并量化
    tokenizer = util.load_tokenizer(tokenizer_path=tokenizer_path)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    quantized_model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config=quantization_config, device_map=device)

    if verbose:
        #量化模型推理
        p = pipeline('text-generation', model=quantized_model, tokenizer=tokenizer)
        print("Quantized predict:")
        print(p("Bitsandbytes is")[0]["generated_text"])

    #保存量化模型
    if tokenizer_save_path is not None:
        tokenizer_save_path = Path(tokenizer_save_path)
        tokenizer.save_pretrained(save_directory=tokenizer_save_path)
        print(f"Tokenizer saved to: {tokenizer_save_path.absolute()}")
    else:
        print("Skip tokenizer saving. No save directory specified.")
    if model_save_path is not None:
        model_save_path = Path(model_save_path)
        quantized_model.save_pretrained(save_directory=model_save_path)
        print(f"Model saved to: {model_save_path.absolute()}")
    else:
        print("Skip model saving. No save directory specified.")    

    #加载量化模型并推理
    if verbose and tokenizer_save_path is not None and model_save_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path,use_fast=False)
        loaded_model = AutoModelForCausalLM.from_pretrained(model_save_path,device_map=device)
        p = pipeline('text-generation', model=loaded_model, tokenizer=tokenizer)
        print("Predict after load:")
        print(p("Bitsandbytes is")[0]["generated_text"])

    print("All the work has been done.")


def main():
    bnb_quantize(model_path="facebook/opt-125m", model_save_path="tmp/bnb", device="cuda:0", verbose=True)

if __name__=="__main__":
    main()