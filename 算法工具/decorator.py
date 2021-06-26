import time
from functools import wraps
from icecream import ic
from inspect import signature

'''
  简单装饰器
'''
def record_run_time(func):
  # 保留传入函数的原始信息和签名
  @wraps(func)
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    ic(f'Fuction {func.__name__} cost {end - start}')
    ic(result)
    return result
  
  return wrapper


import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
logging.root.setLevel(logging.NOTSET)

leveMap = {'debugger':logging.DEBUG,'info':logging.INFO,'warn':logging.WARN}

'''
带参装饰器
'''
def logged(level,name=None,message=None):
  ic(level)
  # 真正的包裹函数
  def decorate(func):
    logname = name if name else func.__name__
    log = logging.getLogger(logname)

    logmsg = message if message else f'Function <{func.__name__}> is running...'

    @wraps(func)
    def wrapper(*args, **kwargs):
      log.log(level, logmsg)
      return func(*args, **kwargs)
    return wrapper
  return decorate

@record_run_time
def func_foo(a,b):
  time.sleep(2)
  return a+b

@logged(level=leveMap['warn'])
def func_bar(a,b):
  time.sleep(2)
  return a+b

if __name__ == '__main__':
  # ic(signature(func_foo))
  # func_foo(10,20)
  ic(func_bar(4,6))