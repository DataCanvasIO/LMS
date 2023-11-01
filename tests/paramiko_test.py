import os
import time

import paramiko

key_path = os.path.expanduser("~") + '/.ssh/id_rsa'

private = paramiko.RSAKey.from_private_key_file(key_path)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='172.20.51.5', username='wuchen', pkey=private)


def dir_list(model_path):
    sftp = client.open_sftp()
    try:
        start = time.time()
        listdir = sftp.listdir_attr(model_path)
        end = time.time()
        print(end - start)
    except FileNotFoundError:
        raise FileNotFoundError
    finally:
        sftp.close()
        client.close()
    files = []
    for f in listdir:
        if not f.filename.startswith('.'):
            files.append({"file_name": f.filename, "is_dir": 1, "last_update_time": f.st_atime * 1000})

    for j in range(len(files)):
        for i in range(1, len(files) - j):
            current = files[i]
            prev = files[i - 1]
            if current['last_update_time'] > prev['last_update_time']:
                files[i] = prev
                files[i - 1] = current
            elif current['last_update_time'] == prev['last_update_time']:
                if current['file_name'] < prev['file_name']:
                    files[i] = prev
                    files[i - 1] = current
    return files


start = time.time()
files0 = dir_list("/home/wuchen/opt-125m")
end = time.time()
print(end - start)
print(files0)
