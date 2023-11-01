from transformers import pipeline


class PipelineLoader:
    @staticmethod
    def load(model_path: str, **kwargs):
        print("hahahahah")
        print("====================")
        if 'device' in kwargs:
            pipe = pipeline(task="text2text-generation",
                            model=model_path,
                            trust_remote_code=True,
                            device=kwargs['device'],
                            model_kwargs={"ignore_mismatched_sizes": True})
        else:
            pipe = pipeline(task="text2text-generation",
                            model=model_path,
                            trust_remote_code=True,
                            model_kwargs={"ignore_mismatched_sizes": True})
        return pipe


if __name__ == '__main__':
    PipelineLoader.load("/Users/dev/Documents/APS/models/huggingface/t5-v1_1-small", learning_type="ddd")
