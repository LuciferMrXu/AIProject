#_*_ coding:utf-8_*_
class Student():
    __students={}
    def __init__(self,name,age,addr):
        if name not in Student.__students:
            Student.__students[name] = {'Age': age,'Address':addr}
            self.__name=name
            self.__age=age
            self.__addr=addr

    @classmethod
    def nums(cls):
        for i in range(3):
            admin=input('请输入二级及以上的管理员权限密码：\n')
            if admin=='000000' or admin=='123456':
                print('当前用户%d位'%(len(cls.__students)))
                break
            else:
                if i==2:
                    print('密码错误三次，该账号已被冻结！')
                else:
                    print('管理员密码错误！')

    @classmethod
    def get_users(cls):
        for i in range(3):
            admin=input('请输入一级管理员权限密码：\n')
            if admin=='123456':
                for key, value in cls.__students.items():
                    print('姓名：%s\n出生年份：%s\n地址：%s'% (key, value['Age'], value['Address']))
                    print()
                break
            else:
                if i==2:
                    print('密码错误三次，该账号已被冻结！')
                else:
                    print('管理员密码错误！')

if __name__ == '__main__':
    张三=Student('张三',2000,'上海')
    李四=Student('李四',1995,'南京')
    王二麻子=Student('王二麻子',1993,'合肥')
    Student.nums()
    Student.get_users()