import os

path = r'advs' # 文件夹路径

files = os.listdir(path) # 获取文件夹中所有文件名

for i, file in enumerate(files):
    # print(type(file))
    if "_" in file:
        old_name = os.path.join(path+'/', file) # 原文件名
        new_name = os.path.join(path, file[0:2]+'.png') # 新文件名
        os.rename(old_name, new_name) # 修改文件名