# -- encoding:utf-8 --
"""
从最原始的磁盘文件中读取出发件人、接收人、发件时间、邮件内容这四个部分的信息
Create by ibf on 2018/10/21
"""

file_paths = ['000', '001']
for file_path in file_paths:
    with open(file_path, 'r', encoding='gb2312', errors='ignore') as file:
        flag = False
        content_dict = {}
        for line in file:
            line = line.strip()
            if line.startswith("From:"):
                content_dict['from'] = line[5:]
            elif line.startswith("To:"):
                content_dict['to'] = line[3:]
            elif line.startswith("Date:"):
                content_dict['date'] = line[5:]
            elif not line:
                flag = True

            if flag:
                if 'content' not in content_dict:
                    content_dict['content'] = line
                else:
                    content_dict['content'] += line
        print(content_dict)
