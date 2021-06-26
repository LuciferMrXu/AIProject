#_*_ coding:utf-8_*_
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


np.random.seed(16)   # 里面可填任意值

n=int(input('请输入n的值：'))

b_values=np.random.normal(loc=-1.0,scale=np.sqrt(10.0),size=n)     # normal()产生正太分布随机数，参数loc均值，scale标准差，size产生的数据量
c_values=np.random.normal(loc=0.0,scale=1.0,size=n)


def SGD(n, b_values, c_values, max_iter=1000, tol=0.00001, alpha=0.01, show_img=True):
    """
    计算最小值时候对应的x和y的值
    :param n: 样本数量
    :param b_values: 样本对应的b值
    :param c_values: 样本对应的c值
    :param max_iter: 最大迭代次数
    :param tol: 当变量小于该值的时候收敛
    :param alpha: 梯度下降学习率
    :return:
    """

    def f1(x, b, c):
        return x ** 2 + b * x + c

    def f(x, b_values, c_values):
        """
        原始函数
        :param x:
        :param b_values:
        :param c_values:
        :return:
        """
        result = 0
        for b, c in zip(b_values, c_values):
            # 遍历所有b和c的组合，这里求均值(防止数据量太大，计算困难)
            result += f1(x, b, c) / n
        return result

    def h1(x, b, c):
        return x * 2 + b

    # 定义变量
    step_channge = 1.0 + tol
    step = 0

    # 给定一个初始的x值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b_values, c_values)

    print("当前参数为:")
    print("b_values={},b的均值为:{}".format(b_values, np.mean(b_values)))
    print("c_values={},c的均值为:{}".format(c_values, np.mean(c_values)))

    # 开始迭代循环
    change_numbers = 0
    y_value_changes = []
    if show_img:
        y_value_changes.append(current_y)
    error_value_changes = []
    while step_channge > tol and step < max_iter:
        """
        在一个迭代次数中(Step)，对m条数据进行遍历，每条样本更新一次模型参数
        """
        print(step)
        random_index = np.random.permutation(n)
        for index in random_index:
            b = b_values[index]
            c = c_values[index]
            # 1. 计算梯度值
            current_d_f = h1(current_x, b, c)
            # 2. 更新参数
            current_x = current_x - alpha * current_d_f
            # 3. 计算更新x之后的y值
            tmp_y = f(current_x, b_values, c_values)
            # 4. 记录y的变换大小、更新次数、更新当前的y值
            step_channge = np.abs(current_y - tmp_y)
            current_y = tmp_y
            change_numbers += 1

            # 添加可视化相关值
            if show_img:
                y_value_changes.append(current_y)
                error_value_changes.append(step_channge)

            # 如果模型效果已经达到最优的情况下，直接退出
            if step_channge < tol:
                break

        # 更新迭代次数
        step += 1

    print("最终迭代的次数:{}, 参数的更新次数:{}, 最终的变化率:{}".format(step, change_numbers, step_channge))
    print("最终结果为:{}---->{}".format(current_x, current_y))

    # 可视化代码（看一下y的变化大小以及函数的变换情况）
    if show_img:
        plt.figure(facecolor='w')
        plt.subplot(1, 2, 1)
        plt.plot(range(change_numbers), error_value_changes, 'r-')
        plt.xlabel('更新次数')
        plt.ylabel('变换大小')
        plt.subplot(1, 2, 2)
        plt.plot(range(change_numbers + 1), y_value_changes, 'g-')
        plt.xlabel('更新次数')
        plt.ylabel('损失函数值')
        plt.suptitle('SGD变换情况可视化')
        plt.show()




if __name__=='__main__':
    SGD(n, b_values, c_values)
