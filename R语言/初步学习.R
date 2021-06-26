(x=c(10,11,13,14))
(x=1:10)# 冒号是生成步长为1的等差数列
(x=seq(5,by=2,length=4))# 从5开始，步长为2，向量长度为4（里面四个元素）
(y=seq(10,by=3,along.with = x))# y向量长度与x向量相同

(y1=rep(x,times=3))# 以x向量为模板，time将整体向量复制3次
(y2=rep(x,each=3))# 以x向量为模板，each将x中的每一个元素复制3次
(y3=rep(x,times=2,each=3)) # time和each可以放在一起用
(y3=rep(x,times=2,each=3,length=17))# 截取y3的头17个元素

(x=vector(mode='logical',length=5))# 创建一个空向量，定义为逻辑型，长度为5

(y1[1:5]);(y1[c(1,2,3,4,5)]);(y1[seq(1,5,1)]) # 下标运算符的操作，分号隔开可以同时执行
(y1)
(y1[-5:-1]) # y1元素中前5个元素不要，后面的元素都要
(which(y1>5))# which()函数返回的是符合条件的下标
(y1[which(y1>5)]) # 将y1元素中大于5的返回

length(y1)
mode(y1)  # 查看y1中每一个元素的数据类型
class(y1) # 查看y1向量整体的数据类型

