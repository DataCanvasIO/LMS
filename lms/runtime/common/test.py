from transformers import pipeline

pipe = pipeline(task="text2text-generation",
                model="/Users/dev/Documents/APS/models/huggingface/t5-v1_1-small",
                trust_remote_code=True,
                model_kwargs={"ignore_mismatched_sizes": True})
print(pipe('hello'))