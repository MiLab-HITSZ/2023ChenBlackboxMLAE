import numpy as np
import os
import argparse
import torch
import torchvision.transforms as transforms
from engine import *
from models import *
from voc import *
import os
os.environ ["CUDA_VISIBLE_DEVICES"] = "0"

path = "/home/klh/DesicionBased/" #文件夹目录
npy= os.listdir("/home/klh/DesicionBased//npy") #得到文件夹下的所有文件名称
npy255= os.listdir("/home/klh/DesicionBased/npy255") #得到文件夹下的所有文件名称
x1 = []
x2 = []
xw= []
xr = []
mean = []

# for i in range(35, 36):
#     inputs = img = np.array(
#         Image.open("/home/klh/DesicionBased/boundary_attack_master/images/GCN_2012/{n}_ori.png".format(n=i)).resize((448, 448)).convert(
#             'RGB')).reshape(
#         (1, 448, 448, 3))
#     adv = np.array(
#         Image.open("/home/klh/DesicionBased/boundary_attack_master/images/GCN_2012/{m}_fin.png".format(m=i)).resize((448, 448)).convert(
#             'RGB')).reshape(
#         (1, 448, 448, 3))
#     raw_array = np.array(img) / 255
#     adv_array = np.array(adv) / 255
#     pur = adv_array - raw_array
#     pur_255 = ((adv_array / 2 + 0.5) * 255) - ((raw_array / 2 + 0.5) * 255)
#
#     np.save('/home/klh/DesicionBased/npy/{a}.npy'.format(a=i), pur)
#     np.save('/home/klh/DesicionBased/npy255/{b}_255.npy'.format(b=i), pur_255)



#
for file in npy:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        test = np.load("/home/klh/DesicionBased/npy/" + file).reshape(-1)
        mean.append(np.mean(np.abs(test)))
        x1.append(np.linalg.norm(np.abs(test), ord=1))
        x2.append(np.linalg.norm(np.abs(test), ord=2))
        xw.append(np.linalg.norm(np.abs(test), ord=np.inf))
for file in npy255:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        test_255 = np.load("/home/klh/DesicionBased/npy255/" + file).reshape(-1)
        xr.append(np.sqrt(np.mean(np.square(test_255))))
print("l2:",np.mean(x2),"l1:",np.mean(x1),"li:",np.mean(xw),"mean:",np.mean(mean),"RMSD",np.mean(xr))




# for file in files: #遍历文件夹
#      if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
#           test = np.load(path+"/"+file).reshape(-1)
#           print(type(test))
#           # x = np.linalg.norm(x=test, ord=1)
#           # print(x)
#           # x2 = np.linalg.norm(x=test, ord=2)
#           # xw = np.linalg.norm(x=test, ord=np.inf)
#           mean.append(np.mean(test))
#           x1.append(np.linalg.norm(x=test, ord=1))
#           x2.append(np.linalg.norm(x=test, ord=2))
#           xw.append(np.linalg.norm(x=test, ord=np.inf))
# #
# #           print(np.mean(x1),np.mean(x2),np.mean(xw),np.mean(mean))
#
# for i in range(1,10):
#      test = np.load('data/voc/VOCdevkit/VOC2007/JPEGImages/00000{j}.jpg'.format(j=i),i,encoding='bytes', allow_pickle=True).reshape(-1)
#      print(type(test))
#      # x = np.linalg.norm(x=test, ord=1)
#      # print(x)
#      # x2 = np.linalg.norm(x=test, ord=2)
#      # xw = np.linalg.norm(x=test, ord=np.inf)
#      mean.append(np.mean(test))
#      x1.append(np.linalg.norm(x=test, ord=1))
#      x2.append(np.linalg.norm(x=test, ord=2))
#      xw.append(np.linalg.norm(x=test, ord=np.inf))
#
#      print(np.mean(x1), np.mean(x2), np.mean(xw), np.mean(mean))
#
#
# raw_image=Image.open("face_scipy.jpg")
# adv_image=Image.open("face_scipy.jpg")
# raw_array=np.array(raw_image)
# adv_array=np.array(adv_image)
#
# np.save('test.npy',adv_array-raw_array)
