import os
from configparser import ConfigParser

import requests


def get_hostname():
    config = ConfigParser()
    config.read(os.path.expanduser("~") + '/.lms/lms.config')

    if config.__contains__('web') and config['web'].__contains__('entrypoint'):
        return config['web']['entrypoint']
    else:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        return ip


def get_server_id():
    config = ConfigParser()
    config.read(os.path.expanduser("~") + '/.lms/lms.config')
    if not config.__contains__('web') or not config['web'].__contains__('server_id'):
        raise Exception("First of all, you should init current client using 'lms join' command.")
    else:
        server_id = config['web']['server_id']
    return server_id


def get_server_host():
    config = ConfigParser()
    config.read(os.path.expanduser("~") + '/.lms/lms.config')
    if not config.__contains__('web') or not config['web'].__contains__('entrypoint'):
        raise Exception("First of all, you should init current client using 'lms join' command.")
    else:
        server_host = config['web']['server']
    return server_host


def query_model_name(model_path):
    if model_path is not None:
        query_url = "http://%s/lms/internal/models/" % get_server_host()
        response = requests.get(url=query_url, headers={}, params={
            "model_path": model_path,
            "hostname": get_hostname()
        })
        if response.status_code == 200:
            models = response.json()['models']
            if len(models) == 1:
                model_name = models[0]['model_name']
                return model_name
            elif len(models) == 0:
                raise Exception("The model:%s is not exist in server" % model_path)
            else:
                raise Exception("duplicated model path")
        else:
            raise Exception(f'{response.status_code}:{response.text}')
    else:
        raise Exception("Both model_path and model_name are none")


def query_model_path(model_name):
    if model_name is not None:
        import re
        if re.match("^[\w_-]+$", model_name) is None:
            raise Exception("The model_name:%s is not a valid name" % model_name)

        query_url = "http://%s/lms/models/%s" % (get_server_host(), model_name)
        response = requests.get(url=query_url, headers={})
        if response.status_code == 200:
            model_path = response.json()['model_path']
            # return model_path[model_path.index(":") + 1:]
            return model_path
        else:
            raise Exception(f'{response.status_code}:{response.text}')
    else:
        raise Exception("model_name is none")


def fill_missing(model_path, model_name):
    if model_name is not None:
        return query_model_path(model_name), model_name
    elif model_path is not None:
        return model_path, query_model_name(model_path)
    else:
        raise Exception("Both model_path and model_name are none")


def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


if __name__ == '__main__':
    print(get_hostname())
