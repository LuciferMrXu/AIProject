class Mystack():
  def __init__(self):
      self.items = []

  # 判断栈是否为空
  def is_empty(self):
    return len(self.items) == 0

  # 获取栈中的元素个数
  def size(self):
    return len(self.items)

  # 返回栈顶元素
  def get_top(self):
    # 先判断栈是否为空，若不为空返回栈顶元素，否则返回None
    if not self.is_empty():
      return self.items[self.size()-1]
    else:
      return None

  # 弹栈
  def my_pop(self):
    # 先判断栈是否为空，若不为空弹出栈顶元素，否则打印栈为空，并返回None
    if not self.is_empty():
      return self.items.pop()
    else:
      print('当前栈为空！')
      return None

  # 压栈
  def my_push(self,item):
    self.items.append(item)

if __name__ == "__main__":
  s1 = Mystack()
  s1.my_push(1)
  print(s1.size())


