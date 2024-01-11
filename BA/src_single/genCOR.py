import pandas as pd
import sys
sys.path.append('../')
import os
file =open('C:/Users/62440/Desktop/孔令浩/研一下实验/UnknownData/data/voc2007/files/VOC2007/classification_test.csv','r')
lines=file.readlines()
lines =lines[1:]
row=[]#定义行数组
for x in lines:
    line=x.split(',') #分割后line是一个列表，line的每一个元素是一个str
    line=line[1:]
    #print(line)
    for word in line:
        num = int(word)
        print(num)





'''df = pd.read_csv('C:/Users/62440/Desktop/孔令浩/研一下实验/UnknownData/data/voc2007/files/VOC2007/classification_test.csv')
print(df)
for i in range (len(df)):
    for j in range(2,22):
        print(df[i][j])'''
