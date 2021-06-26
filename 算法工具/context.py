'''
  上下文管理器
'''
import time
from icecream import ic
from functools import lru_cache

class TimeRecord:
  def __init__(self):
    self._start = 0
    self._end = 0

  def __enter__(self):
    self._start = time.time()

  def __exit__(self,exc_type,exc_value,traceback):
    ic('args:',exc_type,exc_value,traceback)
    self._end = time.time()   
    ic(f'function cost { self._end - self._start }s.')  
      
@lru_cache(maxsize=2**7)
def func(n):
  return 1 if n < 2 else func(n-1) + func(n-2)


if __name__ == '__main__':
  with TimeRecord():
    ic(func(100))