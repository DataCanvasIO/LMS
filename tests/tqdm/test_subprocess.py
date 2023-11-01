import codecs
import io
import subprocess
import sys
import threading
import time

from tests.tqdm.wrapper import TextIOWrapper

process = subprocess.Popen(" ".join(["python3", "/Users/dev/PycharmProjects/lms/tests/tqdm/test1.py"]), shell=True,
                           stderr=subprocess.PIPE,
                           # stdout=subprocess.PIPE,
                           )


def print_progress(stderr):
    # stderr = io.TextIOWrapper(stderr, newline='')
    # while True:
    #     b = stderr.read(1)
    #     sys.stdout.write(b)

    make_decoder = codecs.getincrementaldecoder('utf-8')
    _decoder = make_decoder("strict")
    _CHUNK_SIZE = 512

    while True:
        input_chunk = stderr.read(_CHUNK_SIZE)
        decoded_chars = _decoder.decode(input_chunk, False)
        buf = ''
        for c in decoded_chars:
            if c == '\r':
                # sys.stderr.write('\r')
                sys.stderr.write(buf)
                sys.stderr.write('\r')
                buf = ''
            else:
                buf += c
        # sys.stdout.write(decoder.decode(b, final=True))
        # sys.stdout.write(b.decode("utf-8",'strict'))


t1 = threading.Thread(target=print_progress, args=(process.stderr,))
t1.start()
t1.join()
