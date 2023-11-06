from transformers import pipeline


class PipelineLoader:
    @staticmethod
    def load(model_path: str, **kwargs):
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


