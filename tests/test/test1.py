from tqdm import tqdm
import time
for i in tqdm(range(1000)):
    # 假设我们正在进行一些耗时的操作，比如训练深度学习模型
    time.sleep(0.01)
print("hello")
raise Exception("abcd")