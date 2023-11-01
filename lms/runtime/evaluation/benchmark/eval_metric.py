import random
from typing import List
import evaluate
import numpy as np
import re


class AccEvaluator():
    """Accuracy evaluator."""

    def __init__(self, metric: str = "./accuracy.py", seed: int = 0) -> None:
        self.metric = metric
        self.seed = seed
        super().__init__()

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
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

    def first_capital_postprocess(self, text, index=3, value=["A","B","C","D"]):
        for t in text:
            if t in value:
                return t
        try:
            return value[ord(text.strip()[0])%4]
        except:
            return value[index] 

    def eval(self, predictions: List, references: List) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: calculated scores.
        """
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)
        if len(predictions) != len(references):
            return {
                'error':
                    'predictions and references have different '
                    f'length. len(predictions): {len(predictions)}, '
                    f'len(references): {len(references)}'
            }
        metric = evaluate.load(self.metric)
        predictions = list(map(self.first_capital_postprocess, predictions))
        scores = metric.compute(**self._preprocess(predictions, references))
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        result = {}
        result["acc"] = round(scores['accuracy'],2)
        return result


class MCAccEvaluator():
    """Accuracy evaluator."""

    def __init__(self, metric: str = "./accuracy.py", seed: int = 0) -> None:
        self.metric = metric
        self.seed = seed
        super().__init__()

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
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

    def bbh_mcq_postprocess(self,text, index=0, value=["(A)","(B)","(C)"]) :
        ans = text
        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        match = re.search(r'\(([A-Z])\)*', ans)
        if match:
            return match.group()
        try:
            return value[ord(text.strip()[0])%3]
        except:
            return value[index] 

    def eval(self, predictions: List, references: List) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: calculated scores.
        """
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)
        if len(predictions) != len(references):
            return {
                'error':
                    'predictions and references have different '
                    f'length. len(predictions): {len(predictions)}, '
                    f'len(references): {len(references)}'
            }
        metric = evaluate.load(self.metric)
        predictions = list(map(self.bbh_mcq_postprocess, predictions))
        scores = metric.compute(**self._preprocess(predictions, references))
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        result = {}
        result["acc"] = round(scores['accuracy'],2)
        return result
