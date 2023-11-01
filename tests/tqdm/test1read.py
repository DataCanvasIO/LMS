import locale

from tqdm import tqdm
from time import sleep
import sys

print(locale.getpreferredencoding(False))

sys.stderr.write("hahahah\n")
with open('/Users/dev/PycharmProjects/lms/tests/tqdm/a.txt','r',encoding='UTF-8') as f:
    while True:
        b = f.read(1)
        try:
            # if b == b'\r':
            #     print('\n')
            sys.stdout.write(b)
        except UnicodeDecodeError:
            # print(b)
            # print(b == '\x96')
            # sys.stderr.write('█')
            sys.stderr.write('▍')
    # sys.stderr.write('\r')

#
# sys.stderr.write("hahahah\n")
# for i in tqdm(range(0, 100), desc="Text You Want"): #file=sys.stdout,
#     sleep(.1)