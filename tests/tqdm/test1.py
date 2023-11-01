from tqdm import tqdm
from time import sleep
import sys

# sys.stderr.write("hahahah\n")
# with open('/Users/dev/PycharmProjects/lms/tests/tqdm/a.txt','w') as f:
#     for i in tqdm(range(0, 100),file=f, desc="Text You Want"):
#         sleep(.1)


sys.stderr.write("hahahah\n")
for i in tqdm(range(0, 100), desc="Text You Want"): #file=sys.stdout,
    sleep(.1)
sys.stderr.write("hello\n")