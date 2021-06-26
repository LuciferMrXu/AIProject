from threading import Lock,Thread
from multiprocessing import cpu_count,Pool,freeze_support
import random
import threading
from icecream import ic
from concurrent import futures
import time

from icecream.icecream import DEFAULT_ARG_TO_STRING_FUNCTION

# 顺序获取锁，解决死锁问题
class Acquire:
  def __init__(self, *locks):
      self._locks = sorted(locks, key=lambda x: id(x))

  def __enter__(self):
      for lock in self._locks:
        lock.acquire()

  def __exit__(self,exc_type,exc_value,traceback):
      for lock in reversed(self._locks):
        lock.release()

def philosopher(left,right):
  while True:
    with Acquire(left,right):
    # with Lock():
      ic(f'Thread {threading.currentThread()} is eating...')

def compute(n):
  '''
    cpu密集型任务（用多进程），一般是计算型任务。
    io密集型任务（用多线程），任务不是可以直接完成的，需要等待。如：读文件，下载网页，键盘输入
  '''
  return sum([
      random.randint(1,100) for _ in range(10000)
    ])

# 同步问题
class Account:
  def __init__(self,money=100):
      self.money = money
      self._lock = Lock()

  def save(self,delta):
    # 锁也是一种上下文管理器
    with self._lock:
      self.money += delta

  def withdraw(self,delta):
    with self._lock:
      self.money -= delta

account = Account(500)

def change(n):
    account.save(n)
    account.withdraw(n)

def task(n):
    for _ in range(100000):
        change(n)


if __name__ == '__main__':
  key = 3
  if key == 1:
    # 在主进程中起多个线程
    tasks = [Thread(target=task, args=(money,)) for money in [300,400,500,600,700]]

    for task in tasks:
      task.start()

    for task in tasks:
      task.join()

    ic(account.money)
  elif key == 2:
    # 创建进程池
    pool = Pool(cpu_count())
    ic(f'Result {pool.map(compute,range(11))}')
  elif key == 3:
    chopsticks = [threading.Lock() for _ in range(5)]
    for i in range(5):
      t = threading.Thread(target=philosopher,args=(chopsticks[i],chopsticks[(i+1) % 5]))
      t.start()
      t.join()