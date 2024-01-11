import pandas as pd
import sys
sys.path.append('../')
import os
#gjlb=pd.read_excel('../攻击列表.xls')#这个会直接默认读取到这个Excel的第一个表单
df = pd.read_excel('C:/Users/62440/Desktop/孔令浩/研一下实验/MLDE/src/攻击列表_hidesingle.xls')
#df2 = pd.read_excel('../unsucc_mla_lp.xls')
print(len(df))

def jiaoji (d1 ,d2 ):
    ans = []
    for i in range (len(d1)):
        for j in range (len(d2)):
            if(d1[0][i] == d2[0][j]):
                ans.append(d2[0][j])
                break;
    output = pd.DataFrame(ans)
    output.to_excel('交集.xls')

def trasform (gjlb,df): #将739中的对应元名称翻译成图片名称
    ans = []
    for i in range(len (df)):
        tem = gjlb[0][df[0][i]]
        ans.append(tem)
    output = pd.DataFrame(ans)
    output.to_excel('转化结果.xls')
    return
def show (df):
    ans = []
    for i in range(len(df)):
        ans.append(df[0][i]-1)
    print(ans)

show(df)