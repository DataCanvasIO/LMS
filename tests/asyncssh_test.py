import asyncio, asyncssh, sys
import os
import time

key_path = os.path.expanduser("~") + '/.ssh/id_rsa'
cert_path = os.path.expanduser("~") + '/.ssh/id_rsa.pub'

k = asyncssh.read_private_key(key_path)


# c = asyncssh.read_certificate(cert_path)


async def listdir(path) -> None:
    async with asyncssh.connect('172.20.51.5', username='wuchen',
                                client_keys=[(k, None)], known_hosts=None) as conn:
        async with conn.start_sftp_client() as sftp:
            start = time.time()
            filenames = await sftp.listdir(path)
            files = []
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                attrs = await sftp.lstat(path + "/" + filename)
                files.append(
                    {"file_name": filename, "is_dir": True if attrs.type == 2 else False,
                     "last_update_time": attrs.mtime * 1000})
            end = time.time()
            print(end - start)
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


async def read_file_chunked(path, chunk_size=1024):
    async with asyncssh.connect('172.20.51.5', username='wuchen',
                                client_keys=[(k, None)], known_hosts=None) as conn:
        async with conn.start_sftp_client() as sftp:

            if not await sftp.isfile(path):
                raise Exception("The path is not a file")

            async with await sftp.open(path) as f:
                while True:
                    data = await f.read(chunk_size)
                    if not data:
                        break
                    yield data


try:
    start = time.time()
    files0 = asyncio.get_event_loop().run_until_complete(listdir("/home/wuchen/opt-125m"))
    end = time.time()
    print(end - start)
    print(files0)
except (OSError, asyncssh.Error) as exc:
    sys.exit('SFTP operation failed: ' + str(exc))

# async def main():
#     async for i in read_file_chunked('/home/wuchen/opt-125m/'):
#         print(i)
#
#
# loop = asyncio.get_event_loop()
# try:
#     loop.run_until_complete(main())
# finally:
#     loop.run_until_complete(
#         loop.shutdown_asyncgens())  # see: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.shutdown_asyncgens
#     loop.close()
