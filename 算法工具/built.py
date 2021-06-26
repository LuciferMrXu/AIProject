import math
from icecream import ic
from array import array
import reprlib
from functools import reduce
import itertools

class Vector:
   # N维向量
   typecode = 'd'

   def __init__(self,components):
     self._components = array(self.typecode,components)
 
   # __str__方法在print打印时调用
   def __str__(self):
       return str(tuple(self))
   # __iter__方法在有迭代行为时调用
   def __iter__(self):
       return (i for i in self._components)
   # __repr__方法在直接调用类名时时调用  
   def __repr__(self):
      '''
       return Vector([1.0,2.0,3.0...])
      '''
      components = reprlib.repr(self._components)
      components = components[components.find('['):-1]
      return f'{type(self).__name__}({components})'

   def __hash__(self):
      hash_list = map(lambda x: hash(x), self._components)
      return reduce(lambda a,b: a^b,hash_list,0)

   # 运算符重载，重载__eq__，判断两个向量是否相等
   def __eq__(self,v):
       if len(self) != len(self):
          return False
       else:
          for a,b in zip(self,v):
            if a!=b:
              return False
          return True
    #  return len(self) == len(self) and all(a==b for a,b in zip(self,v))
   
   # 向量取模
   def __abs__(self):
       return math.sqrt(sum( x * x for x in self._components))

   def __bool__(self):
       return bool(abs(self))

   def __len__(self):
       return len(self._components)

   def __getitem__(self,index):
       cls = type(self)
       if isinstance(index,slice):
         return cls(self._components[index])
       elif isinstance(index,int):
         return self._components[index]
       else:
         raise TypeError(f'{cls.__name__} indices must be integers.')


   def __add__(self,v):
      cls = type(self)
      # 两个向量遍历相加计算，长度不等用0填充
      return cls([a+b for a,b in itertools.zip_longest(self,v,fillvalue=0)])

   def __radd__(self,v):
      return self + v

   def __mul__(self,scalar):
      cls = type(self)
      # 向量数乘
      return cls([a*scalar for a in self])

   def __rmul__(self,scalar):
      return self * scalar

   def __matmul__(self,v):
      cls = type(self)
      # 两个向量点乘计算，长度不等用1填充
      return cls([a*b for a,b in itertools.zip_longest(self,v,fillvalue=1)])

   def __rmatmul__(self,v):
      return self @ v


class CalculabilityMixin:

  def plus(self,v):
    cls = type(self)
    # 两个向量遍历相加计算，长度不等用0填充
    return cls([a+b for a,b in itertools.zip_longest(self,v,fillvalue=0)])

  def minus(self,v):
    cls = type(self)
    # 两个向量遍历相减计算，长度不等用0填充
    return cls([a-b for a,b in itertools.zip_longest(self,v,fillvalue=0)])

  def dot(self,v):
    cls = type(self)
    # 两个向量点乘计算，长度不等用1填充
    return cls([a*b for a,b in itertools.zip_longest(self,v,fillvalue=1)])

'''
继承vector
'''
class Vector2d(Vector):
  # 限定属性，使用__slots__方法后，不能在使用__dict__方法获取所有属性，还能节省内存
  __slots__ = ('__x','__y')

  def __init__(self, x, y):
      super().__init__([x, y])
      self.__x = x
      self.__y = y

  @property
  def x(self):
       return self.__x

  @property
  def y(self):
       return self.__y



class VectorOptions(CalculabilityMixin,Vector):
  def __init__(self, components):
      super().__init__(components)

if __name__ == '__main__':
  # v1 = Vector(range(10))
  # ic(v1)
  # print(v1)
  # v2 = Vector2d(5,6)
  # print(dir(v2))
  # ic(v2.x)
  v3 = VectorOptions([2,4,6,7])
  v4 = VectorOptions([4,7])
  print(v3.dot(v4))