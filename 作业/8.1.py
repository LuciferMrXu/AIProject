#_*_ coding:utf-8_*_
class Bank():
    # 定义一个属于银行的类属性
    __Users={}
    def __init__(self,CardID,password,name,balance):
        if CardID not in Bank.__Users:
            Bank.__Users[CardID]={'pwd':password,'UserName':name,'Balance':balance}
            self.__CardID=CardID
            self.__password=password
            self.__name=name
            self.__balance=balance

    # 查看本银行的开户总数
    @classmethod
    def nums(cls):
        for i in range(3):
            admin=input('请输入二级及以上的管理员权限密码：\n')
            if admin=='000000' or admin=='123456':
                print('当前用户%d位'%(len(cls.__Users)))
                break
            else:
                if i==2:
                    print('密码错误三次，该账号已被冻结！')
                else:
                    print('管理员密码错误！')

    # 查看所有用户的个人信息
    @classmethod
    def get_users(cls):
        for i in range(3):
            admin=input('请输入一级管理员权限密码：\n')
            if admin=='123456':
                for key, value in cls.__Users.items():
                    print('卡号：%s\n用户名：%s\n密码：%s\n余额：%.2f' % (key, value['UserName'], value['pwd'], value['Balance']))
                    print()
                break
            else:
                if i==2:
                    print('密码错误三次，该账号已被冻结！')
                else:
                    print('管理员密码错误！')

    #用户验证
    @staticmethod
    def check_User(CardID,password):
        if (CardID in Bank.__Users) and (password==Bank.__Users[CardID]['pwd']):
            return True
        else:
            return False

    # 验证金额
    @staticmethod
    def check_money(money):
        if money.isdigit():
            money=int(money)
            if money>=0 and money%100==0:
                return True
            else:
                return False
        else:
            return False

    # 存取查
    def function(self):
        j=0
        for i in range(3):
            CardID = input('请输入账号：')
            password = input('请输入密码：')
            if Bank.check_User(CardID,password):
                while True:
                    if j == 0:
                        pass
                    elif j == 1:
                        break
                    a = input('请选择：\n1：取款\n2：存款\n3：查询信息\n')
                    if a=='1':
                        money = input('请输入取款金额：')
                        if Bank.check_money(money):
                            if Bank.__Users[CardID]['Balance']>=int(money):
                                Bank.__Users[CardID]['Balance']-=int(money)
                                print('您从卡号为%s的银行卡取款%s元，当前余额%.2f元'%(CardID,money,Bank.__Users[CardID]['Balance']))
                            else:
                                print('余额不足！')
                        else:
                            print('您输入的金额有误！')
                    elif a=='2':
                        money = input('请输入存款金额：')
                        if Bank.check_money(money):
                            Bank.__Users[CardID]['Balance'] += int(money)
                            print('您为卡号为%s的银行卡存款%s元，当前余额%.2f元' % (CardID, money, Bank.__Users[CardID]['Balance']))
                        else:
                            print('您输入的金额有误！')
                    elif a=='3':
                        print('您卡号为%s的银行卡当前余额%.2f元' % (CardID, Bank.__Users[CardID]['Balance']))

                    else:
                        print('请输入正确的数字')
                    while True:
                        b = input('是否继续操作(y/n):\n')
                        if b == 'y':
                            j = 0
                            break
                        elif b == 'n':
                            j = 1
                            break
                        else:
                            print('请重新输入！')
                print('交易完成，请取卡！')
                break
            else:
                if i==2:
                    print('登陆失败，账号已被冻结！')
                else:
                    print('您输入的账号或密码有误！')

if __name__ == '__main__':
    Bob=Bank('12580','88888888','Bob',1235.34)
    Sam=Bank('12306','1234567','Sam',2235.46)
    Lucy=Bank('10086','000000','Lucy',6523.33)
    Bank.nums()
    Bank.get_users()
    Sam.function()