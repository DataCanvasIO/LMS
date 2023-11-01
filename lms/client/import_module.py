import logging
import os.path
import pathlib
import re

import requests

from lms.client.common_utils import get_dir_size, get_server_host, get_hostname
from lms.client.config import hostname
from lms.client.config import server_host

logger = logging.getLogger(__name__)


def gen_model_name(model_path, model_name=None):
    if model_name is None:
        return pathlib.Path(model_path).name
    else:
        return model_name


def add(model_path, model_name):
    if not os.path.exists(model_path):
        raise Exception("The file path:%s isn't exists" % model_path)

    model_path = os.path.abspath(model_path)
    model_name = gen_model_name(model_path, model_name)

    if re.match("^[\w_-]+$", model_name) is None:
        raise Exception("The model_name:%s is not a valid name" % model_name)

    config = None
    config_path = '%s/%s' % (model_path, 'config.json')
    if os.path.isfile(config_path):
        import json
        with open(config_path) as f:
            config = json.load(f)
            print(json.dumps(config, indent=1))
    else:
        logger.warning("The config file:%s isn't exists" % config_path)
    if config is not None:
        quantization_config = config.get("quantization_config", None)
        if quantization_config:
            if quantization_config.get("load_in_8bit", False):
                precision = 'int8'
            elif quantization_config.get("load_in_4bit", False):
                precision = 'int4'
            elif quantization_config.get("bits") == 4:
                precision = 'int4'
            elif quantization_config.get("bits") == 8:
                precision = 'int8'
            else:
                precision = config.get('torch_dtype', 'unknown')
        else:
            precision = config.get('torch_dtype', 'unknown')
    else:
        precision = 'unknown'

    import_url = "http://%s/lms/internal/models" % server_host
    json = {
        "model_path": model_path,
        "hostname": hostname,
        "model_name": model_name,
        "size": get_dir_size(model_path),
        "precision": precision
    }
    response = requests.post(url=import_url, json=json)
    if response.status_code == 200:
        print("Successfully imported %s" % model_name)
    else:
        raise Exception(f'{response.status_code}:{response.text}')


def delete(model_path, model_name):
    model_url = "http://%s/lms/models/%s" % (server_host, model_name)
    response = requests.get(url=model_url, headers={})
    if response.status_code == 404:
        pass
    elif response.status_code == 200:
        if response.json()['status'] == 'deployed':
            raise Exception("The model is deployed. Please undeployment before deleting")
    else:
        pass

    import_url = "http://%s/lms/internal/models/%s" % (server_host, model_name)
    response = requests.delete(url=import_url)
    if response.status_code == 200:
        print("Successfully deleted %s" % model_name)
    else:
        raise Exception(f'{response.status_code}:{response.text}')


def _list():
    query_url = "http://%s/lms/internal/models/" % get_server_host()
    response = requests.get(url=query_url, params={
        "hostname": get_hostname()
    })
    if response.status_code == 200:
        models = response.json()['models']
        return models
    else:
        raise Exception(f'{response.status_code}:{response.text}')
