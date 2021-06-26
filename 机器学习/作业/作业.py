#_*_ coding:utf-8_*_
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

'''
最大似然估计MLE案例（作业）
  变量x: 1,2,3
  变量x出现的概率分布：
    p(1) = 0.5θ
	p(2) = 0.3 + 0.4θ
	p(3) = 0.7 - 0.9θ
  现在随机产生一组序列(有放回的)，序列是：1 2 2 2 1 2 2 3 1 3
'''



'''
梯度下降作业：
  目标函数：
    y = x**2 + b * x + c
  需求：求解最小值对应的x和y
  要去：写代码
    数据：
		b: 服从均值为-1，方差为10的随机数
		c：服从均值为0，方差为1的随机数
	假定a、b、c这样的数据组合总共10、100、10w、100w条数据,求解在现在的数据情况下，目标函数的取最小值的时候，x和y分别对应多少？
'''


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

'''
    随机数据产生
'''
# 设置随机数种子，当程序多次运行时，使每次运行产生的随机数固定为第一次产生的随机数
np.random.seed(16)   # 里面可填任意值

n=int(input('请输入n的值：'))

b_values=np.random.normal(loc=-1.0,scale=np.sqrt(10.0),size=n)     # normal()产生正太分布随机数，参数loc均值，scale标准差，size产生的数据量
c_values=np.random.normal(loc=0.0,scale=1.0,size=n)

# def calc_min_value_with_one_sample(b_values, c_values, max_iter=1000, tol=0.00001, alpha=0.01):
#     """
#     计算最小值时候对应的x和y的值
#     :param b_values: 样本对应的b值
#     :param c_values: 样本对应的c值
#     :param max_iter: 最大迭代次数
#     :param tol: 当变量小于该值的时候收敛
#     :param alpha: 梯度下降学习率
#     :return:
#     """
#
#     # 原始函数
#     def f(x, b, c):
#         return x ** 2 + b * x + c
#
#     # 原始函数对应的导函数
#     def h(x, b):
#         return 2 * x + b
#
#     # 定义变量
#     step_channge = 1.0 + tol
#     step = 0
#
#     # 获取第一个样本
#     b = b_values[0]
#     c = c_values[0]
#
#     # 给定一个初始的x值
#     current_x = np.random.randint(low=-10, high=10)
#     current_y = f(current_x, b, c)
#
#     print("当前参数为:")
#     print("b={}".format(b))
#     print("c={}".format(c))
#
#     GD_X = []
#     GD_Y = []
#     # 开始迭代循环
#     while step_channge > tol and step < max_iter:
#         # 1. 计算梯度值
#         current_d_f = h(current_x, b)
#         # 2. 更新参数
#         current_x = current_x - alpha * current_d_f
#         # 3. 计算更新x之后的y值
#         tmp_y = f(current_x, b, c)
#         # 4. 记录y的变换大小、更新迭代次数、更新当前的y值
#         step_channge = np.abs(current_y - tmp_y)
#         step += 1
#         current_y = tmp_y
#
#         GD_X.append(current_x)
#         GD_Y.append(current_y)
#
#     print("最终更新的次数:{}, 最终的变化率:{}".format(step, step_channge))
#     print("最终结果为:{}---->{}".format(current_x, current_y))
#
#     # 构建数据
#     X = np.arange(-4, 4.5, 0.05)
#     Y = np.array(list(map(lambda t: f(t, b, c), X)))
#
#     # 画图
#     plt.figure(facecolor='w')
#     plt.plot(X, Y, 'r-', linewidth=2)
#     plt.plot(GD_X, GD_Y, 'bo--', linewidth=2)
#     plt.title('函数$y = x^2 + %.3fx + %.3f$; \n学习率:%.3f; 最终解:(%.3f, %.3f);迭代次数:%d' % (b,c,alpha, current_x, current_y, step))
#     plt.show()




# def calc_min_value_with_ten_sample(n,b_values, c_values, max_iter=1000, tol=0.00001, alpha=0.01):
#     """
#     计算最小值时候对应的x和y的值
#     :param n: 样本数量
#     :param b_values: 样本对应的b值
#     :param c_values: 样本对应的c值
#     :param max_iter: 最大迭代次数
#     :param tol: 当变量小于该值的时候收敛
#     :param alpha: 梯度下降学习率
#     :return:
#     """
#     # 要求n必须等于10
#     assert n == 10 and len(b_values) == n and len(c_values) == n
#
#     def f(x, b_values, c_values):
#
#         sample_1 = x ** 2 + b_values[0] * x + c_values[0]
#         sample_2 = x ** 2 + b_values[1] * x + c_values[1]
#         sample_3 = x ** 2 + b_values[2] * x + c_values[2]
#         sample_4 = x ** 2 + b_values[3] * x + c_values[3]
#         sample_5 = x ** 2 + b_values[4] * x + c_values[4]
#         sample_6 = x ** 2 + b_values[5] * x + c_values[5]
#         sample_7 = x ** 2 + b_values[6] * x + c_values[6]
#         sample_8 = x ** 2 + b_values[7] * x + c_values[7]
#         sample_9 = x ** 2 + b_values[8] * x + c_values[8]
#         sample_10 = x ** 2 + b_values[9] * x + c_values[9]
#         return sample_1 + sample_2 + sample_3 + sample_4 + sample_5 + sample_6 + sample_7 + sample_8 + sample_9 + sample_10
#
#     def h(x, b_values):
#
#         sample_1 = x * 2 + b_values[0]
#         sample_2 = x * 2 + b_values[1]
#         sample_3 = x * 2 + b_values[2]
#         sample_4 = x * 2 + b_values[3]
#         sample_5 = x * 2 + b_values[4]
#         sample_6 = x * 2 + b_values[5]
#         sample_7 = x * 2 + b_values[6]
#         sample_8 = x * 2 + b_values[7]
#         sample_9 = x * 2 + b_values[8]
#         sample_10 = x * 2 + b_values[9]
#         return sample_1 + sample_2 + sample_3 + sample_4 + sample_5 + sample_6 + sample_7 + sample_8 + sample_9 + sample_10
#
#     # 定义变量
#     step_channge = 1.0 + tol
#     step = 0
#
#     # 给定一个初始的x值
#     current_x = np.random.randint(low=-10, high=10)
#     current_y = f(current_x, b_values, c_values)
#
#     print("当前参数为:")
#     print("b_values={},b的均值为:{}".format(b_values, np.mean(b_values)))
#     print("c_values={},c的均值为:{}".format(c_values, np.mean(c_values)))
#
#     GD_X = []
#     GD_Y = []
#     # 开始迭代循环
#     while step_channge > tol and step < max_iter:
#         # 1. 计算梯度值
#         current_d_f = h(current_x, b_values)
#         # 2. 更新参数
#         current_x = current_x - alpha * current_d_f
#         # 3. 计算更新x之后的y值
#         tmp_y = f(current_x, b_values, c_values)
#         # 4. 记录y的变换大小、更新迭代次数、更新当前的y值
#         step_channge = np.abs(current_y - tmp_y)
#         step += 1
#         current_y = tmp_y
#
#         GD_X.append(current_x)
#         GD_Y.append(current_y)
#     print("最终更新的次数:{}, 最终的变化率:{}".format(step, step_channge))
#     print("最终结果为:{}---->{}".format(current_x, current_y))
#
#     # 构建数据
#     X = np.arange(-4, 4.5, 0.05)
#     Y = np.array(list(map(lambda t: f(t, b_values, c_values), X)))
#
#     # 画图
#     plt.figure(facecolor='w')
#     plt.plot(X, Y, 'r-', linewidth=2)
#     plt.plot(GD_X, GD_Y, 'bo--', linewidth=2)
#     plt.title(
#         '函数$y = x^2 + %.3fx + %.3f$; \n学习率:%.3f; 最终解:(%.3f, %.3f);迭代次数:%d' % (np.mean(b_values), np.mean(c_values), alpha, current_x, current_y, step))
#     plt.show()






# 批量随机下降法（BGD）
def calc_min_value_with_n_sample(n, b_values, c_values, max_iter=1000, tol=0.00001, alpha=0.01,show_img=True):
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
        result = 0
        for b, c in zip(b_values, c_values):
            # 遍历所有b和c的组合，这里求均值(防止数据量太大，计算困难)
            result += f1(x, b, c) / n
        return result

    def h1(x, b):
        return x * 2 + b

    def h(x, b_values):
        result = 0
        for b in b_values:
            # 遍历求解每个b、c组合对应的梯度值，这里求均值(防止数据量太大，计算困难)
            result += h1(x, b) / n
        return result

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
    y_value_changes = []
    if show_img:
        y_value_changes.append(current_y)     # 添加初始y值
    error_value_changes=[]

    while step_channge > tol and step < max_iter:
        # 1. 计算梯度值
        current_d_f = h(current_x, b_values)
        # 2. 更新参数
        current_x = current_x - alpha * current_d_f
        # 3. 计算更新x之后的y值（重新计算损失函数的值）
        tmp_y = f(current_x, b_values, c_values)
        # 4. 记录y的变换大小、更新迭代次数、更新当前的y值
        step_channge = np.abs(current_y - tmp_y)
        step += 1
        current_y = tmp_y


        # 添加可视化相关值
        if show_img:
            y_value_changes.append(current_y)
            error_value_changes.append(step_channge)

    print("最终更新的次数:{}, 最终的变化率:{}".format(step, step_channge))
    print("最终结果为:{}---->{}".format(current_x, current_y))

    # 可视化代码（看误差变化率以及函数的变换情况）
    if show_img:
        plt.figure(facecolor='w')
        plt.subplot(1,2,1)
        plt.plot(range(step),error_value_changes,'r-')
        plt.xlabel('迭代次数')
        plt.ylabel('变换大小')
        plt.subplot(1,2,2)
        plt.plot(range(step+1),y_value_changes,'g-')
        plt.xlabel('迭代次数')
        plt.ylabel('损失函数值')
        plt.suptitle('MGD变换情况可视化')
        plt.show()




if __name__=='__main__':
    # calc_min_value_with_one_sample(b_values, c_values)
    # calc_min_value_with_ten_sample(n,b_values, c_values)
    calc_min_value_with_n_sample(n, b_values, c_values)

