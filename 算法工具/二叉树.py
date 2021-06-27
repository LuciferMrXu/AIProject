from collections import deque # 队列
'''
二叉树：
    1、每个节点最多有两颗子树
    2、左子树和右子树是有序的，次序不能随便颠倒
    3、即使树中某节点只有一颗子树，也要区分他是左子树还是右子树
'''
class BiNode:
    def __init__(self):
        self.data = None  #节点数据
        self.lchild = None #左子树节点
        self.rchild = None #右子树节点

# 将有序数组转换为二叉树
def array_to_tree(arr,start,end):
    '''
    将有序数组转换为二叉树
        arr:有序数组
        start:数组起点索引
        end:数组结尾索引
    '''
    root = None # 初始化根节点数据
    if end >= start:
        root = BiNode() # 创建实例树
        mid = (start+end+1)//2  # 取中点数据索引
        root.data = arr[mid]  # 树的根节点为数组中间元素
        # 使用递归方式用左半部分数组构建root左子树
        root.lchild = array_to_tree(arr,start,mid-1)
        # 使用递归方式用右半部分数组构建root右子树
        root.rchild = array_to_tree(arr,mid+1,end)
    else:
        root = None
    return root


'''
    中序遍历：按左子树 --> 根节点 --> 右子树的顺序访问（深度优先遍历DFS）
    取数组中间元素作为根节点，将数组分为左右两部分
    对数组左右两部分用递归的方式分别构建左右子树
'''
def print_tree_midorder(root):
    '''
    使用中序遍历的方式打印二叉树内容
        root:二叉树根节点
    '''
    # 先判空
    if root == None:
        return
    # 用递归方式，遍历root节点左子树
    if root.lchild != None:
        print_tree_midorder(root.lchild)

    # 遍历root节点
    print(root.data,end=' ')

    # 用递归方式，遍历root节点右子树
    if root.rchild != None:
        print_tree_midorder(root.rchild)


'''
    层序遍历：按照层从小到大，同一层从左到右依次遍历（广度优先遍历BFS）
    在遍历一个节点的同时记录其子节点的信息，然后按照记录的顺序访问节点数据
    可以采取队列存储当前遍历到的节点的子节点
'''
def print_tree_layer(root):
    '''
    使用层序遍历的方式打印二叉树的内容
        root:二叉树的根节点
    '''
    # 先判空
    if root == None:
        return
    queue = deque() # 创建队列实例
    queue.append(root)

    while len(queue)>0:
        p = queue.popleft() # 出队列
        # 访问当前节点
        print(p.data,end=' ')
        # 如果该节点左子树不为空，则入队列
        if p.lchild != None:
            queue.append(p.lchild)
        # 如果该节点右子树不为空，则入队列
        if p.rchild != None:
            queue.append(p.rchild)


if __name__ == '__main__':
    arr = list(range(1,11))
    root = array_to_tree(arr,0,len(arr)-1)
    print('树的中序遍历结果为：',end=' ')
    print_tree_midorder(root)
    print()
    print('树的层序遍历结果为：',end=' ')
    print_tree_layer(root)