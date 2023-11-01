import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from json import JSONDecodeError

import requests
from requests.exceptions import ProxyError
from urllib3.exceptions import NewConnectionError

from lms.client.config import server_host, hostname, server_id
from lms.client.network_utils import find_free_port, is_port_available
from lms.runtime.common.common_utils import read_infer_config


def is_deployed(model_path):
    cmd = f"ps -ef | grep -Ei 'lms_rt.*deploy.*--model_path={model_path}\s+.*--server_id={server_id}' | grep -v grep | wc -l"
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    if int(result.strip()) > 0:
        return True
    else:
        return False


def deploy(model_path, model_name, port, api_key, timeout, loglevel, unknown, infer_config_path=None, dtype=None,
           infer_py=None, gpu=None):
    """

    :param model_path:
    :param port:
    :return:
    """
    if not os.path.exists(model_path):
        raise Exception("The file path:%s isn't exists" % model_path)

    if is_deployed(model_path):
        raise Exception("The model:%s is already deployed" % model_path)

    if port is None:
        port = find_free_port()
    else:
        if not is_port_available(port):
            raise Exception("The port:%s is already in use" % port)

    from pathlib import Path
    deployment_log_path = Path(os.path.expanduser("~") + "/.lms/logs/deployment")
    deployment_log_path.mkdir(exist_ok=True, parents=True)

    infer_config_pair = "--infer_config=" + infer_config_path if infer_config_path is not None else ""
    command = f"lms_rt deploy --model_path={model_path} --model_name={model_name} --port={port} --server_id={server_id}" \
              f" {'' if dtype is None else '--dtype=' + dtype} " \
              f" {'' if infer_py is None else '--infer_py=' + infer_py} " \
              f" {'' if gpu is None else '--gpu=' + gpu} " \
              f" --api_key={api_key} {infer_config_pair}  --loglevel={loglevel} {' '.join(unknown)} " \
              f" > {deployment_log_path}/{model_name}.log 2>&1 "

    proc = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                            preexec_fn=lambda: os.setpgrp())

    def print_log(_model_name):
        cmd = 'tail -F %s/.lms/logs/deployment/%s.log' % (os.path.expanduser("~"), _model_name)
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        for line in iter(proc.stdout.readline, b''):
            print(str(line, 'utf-8'), end='')

    t1 = threading.Thread(target=print_log, args=(model_name,), daemon=True)
    t1.start()

    start_time = time.time()
    try:
        while True:
            try:
                response = requests.get(url="http://%s:%s/ready" % (hostname, port))
                if response.status_code == 200:
                    break
                elif response.status_code == 503:
                    resp_text = '' if response.text is None else response.text
                    try:
                        import json as jzon
                        result = jzon.loads(resp_text)
                        if 'message' in result:
                            print("Wait please, exec probe response:" + result['message'])
                        else:
                            print("Wait please, exec probe response:" + resp_text)
                    except JSONDecodeError:
                        print("Wait please, exec probe response:" + resp_text)
                else:
                    print(response.status_code)
            except (ConnectionError, NewConnectionError, ProxyError) as e:
                print("Wait please, exec probe response:" + str(e))
            except Exception:
                traceback.print_exception(*sys.exc_info())
            finally:
                poll = proc.poll()
                if poll is not None:
                    raise Exception("Deployment failed")

            if time.time() >= start_time + timeout:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                print("You could use --timeout parameter to prolong the timeout time")
                raise Exception("waiting timeout")
            time.sleep(1)
    except KeyboardInterrupt as e:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        sys.exit(130)

    deploy_url = "http://%s/lms/internal/models/%s/deployment" % (server_host, model_name)
    request_body = {
        "api_url": "http://%s:%s/prediction" % (hostname, port),
        "api_key": api_key
    }

    if infer_config_path is not None:
        infer_config = read_infer_config(infer_config_path)
        request_body['generate'] = infer_config.get('generate', {})

    response = requests.post(url=deploy_url, json=request_body, headers={})
    if response.status_code == 200:
        print("Successfully deployed %s, api_key:%s" % (model_name, api_key))
    else:
        raise Exception(f'{response.status_code}:{response.text}')


def undeploy(model_path, model_name):
    """

    :param model_path:
    :return:
    """
    cmd = f"ps -ef | grep -Ei 'lms_rt.*deploy.*--model_path={model_path}.*--server_id={server_id}|transformers_infer\.py.*--model_path={model_path}.*--server_id={server_id}|deepspeed_infer\.py.*--model_path={model_path}.*--server_id={server_id}' " + \
          " | grep -v grep | awk '{print $2}' | xargs -I {} kill -SIGTERM {} "
    import os
    os.system(cmd)

    from lms.client.config import server_host
    deploy_url = "http://%s/lms/internal/models/%s/deployment" % (server_host, model_name)
    import requests
    header = {}
    response = requests.delete(url=deploy_url, headers=header)
    if response.status_code == 200:
        print("Successfully undeployed %s" % model_name)
    else:
        raise Exception(f'{response.status_code}:{response.text}')


def logs(model_path, model_name, follow):
    cmd = 'tail %s -n +0 %s/.lms/logs/deployment/%s.log' % (
        '-F' if follow else '', os.path.expanduser("~"), model_name)
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    try:
        for line in iter(proc.stdout.readline, b''):
            print(str(line, 'utf-8'), end='')
    except KeyboardInterrupt:
        sys.exit(130)
