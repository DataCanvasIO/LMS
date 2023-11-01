def load_tokenizer(tokenizer_path):
    def try_load(clz):
        try:
            tokenizer = clz.from_pretrained(tokenizer_path)
        except Exception as e:
            print(e)
            print("Failed to load tokenizer, try trust_remote_code=True.")
            tokenizer = clz.from_pretrained(tokenizer_path, trust_remote_code=True)
        return tokenizer

    if "llama" in tokenizer_path.lower():
        from transformers import LlamaTokenizer
        return try_load(LlamaTokenizer)
    else:
        from transformers import AutoTokenizer
        return try_load(AutoTokenizer)