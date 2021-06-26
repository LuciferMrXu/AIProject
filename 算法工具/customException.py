'''
自定义异常
'''
class NetworkError(Exception):
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.args = args
      self.kwargs = kwargs

  def method(self):
      pass


class HostnameError(Exception):
  # if not vaild(hostname):
  #   raise HostnameError('host name is invalid.')
  pass

class TimeoutError(Exception):
  pass

class ProtocolError(Exception):
  pass

