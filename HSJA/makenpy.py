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

# path = "/home/czj/jjr/HSJA-master/tem/" #文件夹目录
# files= os.listdir(path) #得到文件夹下的所有文件名称
# x1 = []
# x2 = []
# xw= []
# mean = []

img = Image.open('data/voc/VOCdevkit/VOC2007/JPEGImages/000010.jpg').resize((448,448)).convert('RGB')
adv = Image.open('advs3w/10-p.png').resize((448,448)).convert('RGB')
raw_array=np.array(img) / 255
adv_array=np.array(adv) / 255
pur = adv_array-raw_array
pur_255 = ((adv_array / 2 + 0.5) * 255) - ((raw_array / 2 + 0.5) * 255)
print(pur_255)

print(pur)
np.save('test10.npy',pur)
np.save('test10_255.npy',pur_255)


print(type(adv))
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
#
#           print(np.mean(x1),np.mean(x2),np.mean(xw),np.mean(mean))

for i in range(1,9):
     img = Image.open('data/voc/VOCdevkit/VOC2007/JPEGImages/00000{j}.jpg'.format(j=i),i).resize((448, 448)).convert('RGB')
     print(type(img))
     imgori = Image.open('data/voc/VOCdevkit/VOC2007/JPEGImages/00000{j}.jpg'.format(j=i),i).resize((448, 448)).convert('RGB')
     # ae = Image.open('advs3w/{j}-p.jpg'.format(j=i),i).resize((448, 448)).convert('RGB')
     print(type(imgori))

     # print(type(test))
     # # x = np.linalg.norm(x=test, ord=1)
     # # print(x)
     # # x2 = np.linalg.norm(x=test, ord=2)
     # # xw = np.linalg.norm(x=test, ord=np.inf)
     # mean.append(np.mean(test))
     # x1.append(np.linalg.norm(x=test, ord=1))
     # x2.append(np.linalg.norm(x=test, ord=2))
     # xw.append(np.linalg.norm(x=test, ord=np.inf))
     #
     # print(np.mean(x1), np.mean(x2), np.mean(xw), np.mean(mean))


# raw_image=Image.open("face_scipy.jpg")
# adv_image=Image.open("face_scipy.jpg")
# raw_array=np.array(raw_image)
# adv_array=np.array(adv_image)
#
# np.save('test.npy',adv_array-raw_array)
