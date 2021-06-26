#_*_ coding:utf-8_*_
import numpy as np
import abc

class Shape(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def circumference(self):
        pass

    @abc.abstractmethod
    def area(self):
        pass


class Rectangle(Shape):
    def __init__(self,long,wide):
        self.long=long
        self.wide=wide

    def circumference(self):
        C=(self.long+self.wide)*2
        return C

    def area(self):
        S=self.long*self.wide
        return S


class Cycle(Shape):
    def __init__(self,radius):
        self.radius=radius

    def circumference(self):
        C=2*self.radius*np.pi
        return C

    def area(self):
        S=np.pi*self.radius**2
        return S


if __name__ == '__main__':
    cycle1=Cycle(3)
    rectangle1=Rectangle(2,4)
    print(cycle1.circumference())
    print(cycle1.area())
    print(rectangle1.circumference())
    print(rectangle1.area())





