#_*_ coding:utf-8_*_
class Class():
    students={}
    def __init__(self,name,age,sex,phone,address):
        if name not in Class.students:
            Class.students[name]={'出生年份':age,'性别':age,'电话号码':phone,'家庭地址':address}
            self.name=name
            self.age=age
            self.sex=sex
            self.__phone=phone
            self.__address=address

    @staticmethod
    def get_student():
        for key, value in Class.students.items():
            print(u"姓名：%s\n出生年份：%s\n性别：%s\n电话号码：%s\n家庭住址：%s" % (key, value['出生年份'], value['性别'], value['电话号码'], value['家庭地址']))
            print()


    @classmethod
    def del_student(cls):
        for i in range(3):
            admin=input('请输入管理员密码：\n')
            if admin=='123456':
                delete=input('请输入删除学员的姓名：\n')
                if delete in cls.students:
                    del cls.students[delete]
                    print('%s学员删除成功！'%delete)
                else:
                    print('查无此人！')
                break
            else:
                if i==2:
                    print('密码错误三次，该账号已被冻结！')
                else:
                    print('管理员密码错误！')

    @classmethod
    def alter_student(cls):
        for i in range(3):
            admin=input('请输入管理员密码：\n')
            if admin=='123456':
                try:
                    find=input('请输入修改信息的学员姓名：\n')
                    print(cls.students[find])
                except Exception:
                    print('查无此人！')
                else:
                    find_one=input('请输入需要修改的信息：\n')
                    if find_one in cls.students[find]:
                        alter=input('请输入修改内容：\n')
                        cls.students[find][find_one]=alter
                    else:
                        print('没有该项信息！')
                finally:
                    break
            else:
                if i==2:
                    print('密码错误三次，该账号已被冻结！')
                else:
                    print('管理员密码错误！')


if __name__ == '__main__':
    n = 0
    while True:
        if n == 0:
            pass
        elif n == 1:
            break
        a = input('请输入姓名：\n')
        b = input('请输入出生年份：\n')
        c = input('请输入性别：\n')
        d = input('请输入电话号码：\n')
        e=input('请输入家庭住址：\n')
        Class(a, b, c, d, e)
        while True:
            m = input('是否继续输入(y/n):\n')
            if m == 'y':
                break
            elif m == 'n':
                n = 1
                break
            else:
                print('请重新输入！')

    Class.get_student()
    Class.del_student()
    Class.get_student()
    Class.alter_student()
    Class.get_student()
