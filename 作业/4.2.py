#_*_ coding:utf-8_*_
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client.mydb
collection=db.学员信息


student1={
    '姓名':'张三','年龄':22,'性别':'男','学号':'119'
}

student2 = {
    '姓名':'李四','年龄':20,'性别':'男','学号':'120'
}


result = collection.insert_many([student1, student2])
print(result)
print(result.inserted_ids)

condition = {'姓名': '李四'}
student = collection.find_one(condition)
student['年龄'] = 26
result = collection.update_one(condition, {'$set': student})
print(result)
print(result.matched_count, result.modified_count)



condition = {'年龄': {'$gt': 20}}
result = collection.update_many(condition, {'$inc': {'年龄': 1}})
print(result)
print(result.matched_count, result.modified_count)


result = collection.delete_one({'姓名': '张三'})
print(result)
print(result.deleted_count)
result = collection.delete_many({'年龄': {'$lt': 25}})
print(result.deleted_count)



