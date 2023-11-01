import asyncio
import codecs
import contextlib
import sys


async def read_stdout(stderr):
    print("hello")
    print(type(stderr))

    make_decoder = codecs.getincrementaldecoder('utf-8')
    _decoder = make_decoder("strict")
    _CHUNK_SIZE = 128

    buf = ''
    while True:
        input_chunk = await stderr.read(_CHUNK_SIZE)
        decoded_chars = _decoder.decode(input_chunk, False)
        for c in decoded_chars:
            if c == '\n':
                sys.stderr.write(buf + '\n')
                buf = ''
            elif c == '\r':
                sys.stderr.write('\r')
                sys.stderr.write(buf)
                buf = ''
            else:
                buf += c


async def check_status(proc):
    while True:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), 1e-6)
        await asyncio.sleep(3)
        print("hahah")
        if proc.returncode is not None:
            raise Exception("The Inference process was exited unexpectedly")
        else:
            continue


async def run1():
    infer_args = ["/Users/dev/PycharmProjects/lms/tests/tqdm/test1.py"]
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        *infer_args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.gather(
        read_stdout(proc.stderr),
        check_status(proc)
    )


asyncio.run(run1())
