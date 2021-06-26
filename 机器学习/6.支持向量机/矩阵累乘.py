import numpy as np

if __name__ == '__main__':
    x1 = np.arange(9.0).reshape((3, 3))
    print(x1)
    x2 = np.arange(3.0)
    print(x2)
    c = np.multiply(x1, x2)
    print(c)