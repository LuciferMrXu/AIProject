# -*- coding: utf-8 -*-
'''
编辑距离：从一个字符串变成另一个字符串，增、删、改、查，每进行一步编辑距离加一
'''
def edits(word1,word2):
    letters='abcdefghijklmnopqrstuvwxyz'
    splits=[(word1[:i],word1[i:]) for i in range(len(word1)+1)]
    print(splits)
    deletes=[L+R[1:] for L,R in splits if R]
    print(deletes)
    transposes=[L+R[1:]+R[0]+R[2:] for L,R in splits if len(R)>1]
    print(transposes)
    replaces=[L+c+R[1:] for L,R in splits if R for c in letters]
    print(replaces)
    inserts=[L+c+R for L,R in splits for c in letters]
    print(inserts)
    vol_value=set(deletes+transposes+replaces+inserts)
    print(vol_value)
    value=[e2 for e1 in vol_value for e2 in vol_value]
    print(len(value))



def edit(str1, str2):        
    matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]      
    for i in range(1,len(str1)+1):       
        for j in range(1,len(str2)+1):           
            if str1[i-1] == str2[j-1]:                
                d = 0            
            else:                
                d = 1            
            matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)      
    value = matrix[len(str1)][len(str2)]  
    print('编辑距离为%s'%value)


if __name__=='__main__':
    a=input('请输入原字符串：\n')
    b=input('请输入对比字符串：\n')
    edit(a,b)
    