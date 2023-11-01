import os

import asyncssh

key_path = os.path.expanduser("~") + '/.ssh/id_rsa'
cert_path = os.path.expanduser("~") + '/.ssh/id_rsa.pub'

k = asyncssh.read_private_key(key_path)


async def listdir(path, node) -> None:
    """
    Read the names of the files in a remote directory
    :param path:
    :param node:
    :return:
    """
    async with asyncssh.connect(node.hostname, username=node.username,
                                client_keys=[(k, None)], known_hosts=None) as conn:
        async with conn.start_sftp_client() as sftp:
            filenames = await sftp.listdir(path)
            files = []
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                attrs = await sftp.lstat(path + "/" + filename)
                files.append({"file_name": filename, "is_dir": True if attrs.type == 2 else False,
                              "last_update_time": attrs.mtime * 1000})
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


async def read_file_chunked(path, node, chunk_size=1024):
    """
    Read the data chunks from a remote file
    :param path:
    :param node:
    :param chunk_size:
    :return:
    """
    async with asyncssh.connect(node.hostname, username=node.username,
                                client_keys=[(k, None)], known_hosts=None) as conn:
        async with conn.start_sftp_client() as sftp:

            if not await sftp.isfile(path):
                raise Exception("The path is not a file")

            async with await sftp.open(path, pflags_or_mode='rb') as f:
                while True:
                    data = await f.read(chunk_size)
                    if not data:
                        break
                    yield data


async def write_file(path, node, data):
    """
    Write the data bytes into a remote file
    :param path:
    :param node:
    :param data:
    :return:
    """
    async with asyncssh.connect(node.hostname, username=node.username,
                                client_keys=[(k, None)], known_hosts=None) as conn:
        async with conn.start_sftp_client() as sftp:
            if await sftp.lexists(path) and not await sftp.isfile(path):
                raise Exception("The path is a dir")
            async with await sftp.open(path, pflags_or_mode='wb') as f:
                await f.write(data)
