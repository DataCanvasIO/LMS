import csv
from typing import List

import evaluate
from datasets import Dataset


class CustomDataset:

    @staticmethod
    def load(data_path: str):
        pre_prompt = "以下是单项选择题，请直接给出正确答案的选项。\n题目："
        post_prompt: str = "答案是："
        raw_data = []
        with open(data_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            _ = next(reader)  # skip the header
            for row in reader:
                assert len(row) == 6
                question = row[0]
                A = row[1]
                B = row[2]
                C = row[3]
                D = row[4]
                raw_data.append({
                    'question': pre_prompt + f"{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}" + post_prompt,
                    'answer': row[5],
                })
        dataset = Dataset.from_list(raw_data)
        return dataset

    @staticmethod
    def eval(predictions: List, references: List, accuracy_script_path) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: calculated scores.
        """
        metric = evaluate.load(accuracy_script_path)

        def first_capital_postprocess(text):
            for t in text:
                if t.isupper():
                    return t
            return ''

        def preprocess(predictions, references):
            mapping_to_int_dict = {
                label: idx
                for idx, label in enumerate(set(map(str, references)))
            }
            pred_set = set(predictions)
            for pred in pred_set:
                if str(pred) not in mapping_to_int_dict.keys():
                    mapping_to_int_dict[str(pred)] = len(mapping_to_int_dict)
            golds = [mapping_to_int_dict[str(gold)] for gold in references]
            preds = [mapping_to_int_dict[str(pred)] for pred in predictions]
            return {
                'predictions': preds,
                'references': golds,
            }

        predictions = list(map(first_capital_postprocess, predictions))
        scores = metric.compute(**preprocess(predictions, references))
        result = {}
        result["acc"] = round(scores['accuracy'], 2)
        return result
