import asyncio

import asyncssh as asyncssh


async def aa():
    path = "/tmp/abc"
    data = "haha\n1234"
    async with asyncssh.connect('127.0.0.1', username='dev', password='123456', known_hosts=None) as conn:
        async with conn.start_sftp_client() as sftp:
            if await sftp.listdir(path):
                raise Exception("The path is a dir")
            async with await sftp.open(path, pflags_or_mode='w') as f:
                await f.write(data)


async def bb(path, chunk_size=2048):
    async with asyncssh.connect('172.20.51.5', username='wuchen', password='wuchen', known_hosts=None) as conn:
        async with conn.start_sftp_client() as sftp:

            if not await sftp.isfile(path):
                raise Exception("The path is not a file")

            async with await sftp.open(path, pflags_or_mode='rb') as f:
                while True:
                    data = await f.read(chunk_size)
                    if not data:
                        break
                    yield data


async def abc():
    async for i in bb("/home/wuchen/human/target11.csv"):
        print(i)

asyncio.run(abc())
