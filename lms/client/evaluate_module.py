import subprocess
import threading
from pathlib import Path

import requests

from lms.client.config import server_host, hostname


def evaluate(model_path, model_name, args, argv):
    # 检查模型是否导入
    command = ['lms_rt']
    command.extend(argv[1:])

    for index, segment in enumerate(command):
        if segment == "--model_path" or \
                segment == "--model_name":
            command[index] = "--model_path"
            command[index + 1] = model_path
            break
        elif segment.startswith("--model_path") or \
                segment.startswith("--model_name"):
            command[index] = "--model_path=" + model_path
            break
        else:
            continue

    process = subprocess.Popen(" ".join(command), shell=True,
                               stdout=subprocess.PIPE)

    def print_progress(stdout):
        from tqdm import tqdm
        tqdm.write(stdout.readline().decode("utf-8"))

    t1 = threading.Thread(target=print_progress, args=(process.stdout,))
    t1.start()
    t1.join()

    for line in iter(process.stdout.readline, b''):
        print(str(line, 'UTF-8'))

    out, err = process.communicate()
    if process.returncode == 0:
        # last_line = str(out, 'UTF-8').splitlines()[-1]
        last_line = line
        # print(last_line)
        import json
        res = json.loads(last_line)

        # if res['eval_kind'] == "custom":
        #     if res.get('benchmarks', None) is None or \
        #             res.get('benchmarks'):
        #         raise Exception("")

        request_body = {
            "model_name": model_name,
            "eval_kind": res['eval_kind'],
            "input_data_path": "%s:%s" % (hostname, args.input_path),
            "output_data_path": "%s:%s" % (hostname, Path(args.output_path).absolute().as_posix()),
            "benchmarks": res.get('benchmarks', None)
        }
        eval_url = "http://%s/lms/internal/models/%s/evaluation" % (server_host, model_name)
        response = requests.post(url=eval_url, json=request_body)

        if response.status_code == 200:
            print("Successfully evaluated %s" % model_name)
        else:
            raise Exception(f'{response.status_code}:{response.text}')
    else:
        # raise Exception(str(out) + str(err))
        exit(process.returncode)
