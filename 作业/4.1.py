#_*_ coding:utf-8_*_
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client.mydb
collection=db.学员信息

n=1
m=0
while True:
    if m == 0:
        pass
    elif  m == 1:
        break
    a = input('请输入学员姓名：')
    b = input('请输入学员年龄：')
    c = input('请输入学员性别：')
    d = input('请输入学员学号：')
    data={
        '序号':n,'姓名':a,'年龄':b,'性别':c,'学号':d
    }
    n+=1
    print(data)
    collection.insert_one(data)
    while True:
        e = input('是否继续输入(y/n):\n')
        if e == 'y':
            break
        elif e == 'n':
            m = 1
            break
        else:
            print('请重新输入！')