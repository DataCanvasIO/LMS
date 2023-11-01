import asyncio
import sys


async def read_stdout(stdout):
    while True:
        line = await stdout.readline()
        if not line:
            break
        print(str(line, 'utf-8').replace('prompt:', ''))


async def read_stderr(stderr):
    while True:
        line = await stderr.readline()
        if not line:
            break
        sys.stderr.write(str(line, 'utf-8'))


async def write_stdin(stdin):
    for i in range(100):
        buf = f'hello {i}\n'.encode()
        stdin.write(buf)
        await stdin.drain()
        await asyncio.sleep(0.5)


async def dd():
    proc = await asyncio.create_subprocess_shell(
        'python3 /Users/dev/PycharmProjects/lms/lms/runtime/deploy/infer.py --model_path=/Users/dev/Documents/APS/models/huggingface/t5-v1_1-small',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    await asyncio.gather(
        read_stderr(proc.stderr),
        read_stdout(proc.stdout),
        write_stdin(proc.stdin))


asyncio.run(dd())
