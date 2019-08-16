from multiprocessing.pool import ThreadPool
import multiprocessing
import time
import threading
from tqdm import tqdm

bars = [tqdm(total=10) for i in range(5)]
for j in range(10):
    for i in range(5):
        bars[i].update(j)
    time.sleep(0.2)
# bar = tqdm(total=10)
# for i in range(10):
#     bar.update(i)
#     time.sleep(0.1)