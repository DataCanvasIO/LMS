import os
import re
import signal
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path

import requests

from lms.client.network_utils import is_port_available, find_free_port

authorized_keys_path = os.path.expanduser("~") + '/.ssh/authorized_keys'

DEFAULT_DEAMON_PORT = 8082


def until_started(log_path):
    cmd = f'tail -F {log_path}'
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    for line in iter(proc.stdout.readline, b''):
        line = str(line, 'utf-8')
        if line == 'started\n':
            print("successful to start lmsd")
            proc.terminate()
            return True
        print(line, end='')
    return False


def join(server, token, entrypoint):
    # restart lmsd if it's running
    pid_path = f'{os.path.expanduser("~")}/.lms/lmsd.pid'
    if os.path.isfile(pid_path):
        with open(pid_path, "r") as f:
            pid = f.read()
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            print("successful to stop lmsd")
        except ProcessLookupError as e:
            print(f'The process:{pid} is not found')
        finally:
            os.remove(pid_path)

    daemon_port = DEFAULT_DEAMON_PORT if is_port_available(DEFAULT_DEAMON_PORT) else find_free_port()

    log_path = f'{os.path.expanduser("~")}/.lms/logs/lmsd.log'

    cmd = f"{sys.executable} -u -m lms.client.daemon.main --port={daemon_port} >{log_path} 2>&1 "
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            preexec_fn=lambda: os.setpgrp())

    if until_started(log_path):
        with open(pid_path, "w") as f:
            f.write(str(proc.pid))
    else:
        if proc.poll() is not None:
            print(f"failed to start log_path with exit code:{proc.returncode}")
            sys.exit(1)

    eval_url = "http://%s/admin/pubkey" % server
    response = requests.post(url=eval_url, json={}, headers={})
    public_key = response.json()['pubkey']
    server_id = response.json()['server_id']
    flag = False

    if os.path.exists(authorized_keys_path):
        Path(authorized_keys_path).touch(mode=600)

    with open(authorized_keys_path, "r") as f:
        for line in f:
            match = re.match(r"[0-9a-zA-Z-]*\s[0-9a-zA-Z+=/]*(\s(?P<comment>.*))?", line)
            if match:
                comment = match.group("comment")
                if comment == 'lms_web':
                    flag = True
                    break
    if not flag:
        with open(authorized_keys_path, 'a') as f:
            match = re.match(r"[0-9a-zA-Z-]*\s[0-9a-zA-Z+=/]*", public_key)
            if match:
                f.writelines([match.group(0) + ' ' + 'lms_web'])
            else:
                raise Exception('invalid pub keys')

    # 加入到web中
    # TODO 利用token加密发送
    join_url = "http://%s/admin/join" % server
    import getpass
    username = getpass.getuser()
    if entrypoint is None:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        answer = input("The HOSTNAME was detected as %s. Are you agree with this?: [Y/n]" % ip)
        if not answer or answer == '' or answer[0].lower() == 'y':
            entrypoint = ip
        else:
            answer = input("Enter the right HOSTNAME:")
            if answer:
                entrypoint = answer
            else:
                answer = input("Malformed HOSTNAME, Enter the right HOSTNAME again:")
                if answer:
                    entrypoint = answer
                else:
                    sys.exit(1)

    json = {
        "hostname": entrypoint,
        "port": 22,
        "username": username,
        "daemon_port": daemon_port
    }
    response = requests.post(url=join_url, json=json, headers={})
    if response.status_code != 200:
        raise Exception(f'{response.status_code}:{response.text}')

    config_path = os.path.expanduser("~") + "/.lms"
    os.makedirs(config_path, exist_ok=True)

    config = ConfigParser()
    config.add_section('web')  # 添加table section
    config.set('web', 'server', server)
    config.set('web', 'server_id', server_id)

    if entrypoint is not None:
        config.set('web', 'entrypoint', entrypoint)
    with open(os.path.expanduser("~") + "/.lms/lms.config", 'w', encoding='utf-8') as file:
        config.write(file)  # 值写入配置文件
    print("Successfully joined")


def reset():
    from lms.client.import_module import _list
    models = _list()
    if len(models) > 0:
        raise Exception('Please delete all models on this node.')

    pid_path = f'{os.path.expanduser("~")}/.lms/lmsd.pid'
    if os.path.isfile(pid_path):
        with open(pid_path, "r") as f:
            pid = f.read()
        os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        os.remove(pid_path)
        print("successful to stop lmsd")
    else:
        print("The lmsd isn't running")

    from lms.client.config import server_host
    url = "http://%s/admin/reset" % server_host
    import getpass
    username = getpass.getuser()
    import socket
    ip = socket.gethostbyname(socket.gethostname())

    json = {
        "hostname": ip,
        "port": 22,
        "username": username
    }
    response = requests.post(url=url, json=json, headers={})
    if response.status_code != 200:
        raise Exception(f'{response.status_code}:{response.text}')

    with open(authorized_keys_path, "r") as fp:
        lines = fp.readlines()

    with open(authorized_keys_path, "w") as fp:
        for line in lines:
            match = re.match(r"[0-9a-zA-Z-]*\s[0-9a-zA-Z+=/]*(\s(?P<comment>.*))?", line)
            if match:
                comment = match.group("comment")
                if comment != 'lms_web':
                    fp.write(line)

    os.remove(os.path.expanduser("~") + "/.lms/lms.config")
    print("Successfully reset")
