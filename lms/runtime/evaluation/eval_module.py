import os
import re


def evaluate1(model_path, task, input_path, output_path):
    if task == 'human':
        from lms.runtime.evaluation.manual_eval import manual_eval
        manual_eval(model_path, input_path, output_path)
        # save to web
        output = {'eval_kind': "human"}
        return output
    elif re.match(r"^.*\.py$", task):
        from lms.runtime.evaluation.custom.eval import do_eval
        accuracy_script_path = os.path.dirname(__file__) + "/accuracy.py"
        output = do_eval(task, input_path, model_path, output_path, accuracy_script_path)
        output['eval_kind'] = "custom"
        return output
    else:
        from lms.runtime.evaluation.benchmark.eval import do_eval, task_map
        tasks = task.split(",")
        check = any(item in task_map.keys() for item in tasks)
        if check is True:
            output = do_eval(task, model_path, output_path)
            output['eval_kind'] = "automatic"
            return output
        else:
            print("No, List1 doesn't have any elements of the List2.")
