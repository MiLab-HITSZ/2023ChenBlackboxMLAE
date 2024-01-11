#@Time      :2021/5/13 18:56
# @Author    :Klein
# @FileName  :ml_de.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import sys
sys.path.append('../yolo_master')
import pandas as pd
import  xml.dom.minidom
import numpy as np
import logging
import torch
import gc
from multiprocessing import Pool
from ml_liw_model.train  import criterion
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from enum import Enum
from yolo_master.yolo_voc2012 import *

class MLDE(object):
    def __init__(self, model):
        self.model = model

    def generate_np(self, x_list, ori_loader, **kwargs):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        logging.info('prepare attack')
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']
        eps = kwargs['eps']
        pop_size = kwargs['pop_size']
        generation = kwargs['generation']
        batch_size = kwargs['batch_size']
        yolonet = kwargs['yolonet']
        x_adv = [] #一张一张保存生成的对抗样本
        success = 0
        nchannels,img_rows, img_cols,  = x_list.shape[1:4]
        count = 0
        l2_sum=0
        diedai = 0
        fail = []
        np.random.seed(1)
        for i in range(len(x_list)):
                # i=8
                best_l2=999
                chance=50 #每张对抗样本在初始化是时尝试50次，一开始门槛设置为100，初始扰动大于100则直接舍弃，10次后如果仍大于100，则将门槛+25，50次后取消门槛设定
                try_time=0
                giveup1 = 0  # 是否跳过yolo拼接
                giveup2 = 1  # 是否跳过数据集拼接
                threshold = 100
                adjust = 10
                isblack = 0
                Adv, isAder, giveup1, giveup2 = GenBig(img_rows * img_cols * nchannels, self.model, x_list[i],
                                                       y_target[i], ori_loader, try_time, yolonet, giveup1, giveup2,
                                                       isblack)

        return x_adv , diedai

class Problem:
    def __init__(self, model, adv_img, target, eps, batch_size, ori_img):
        self.model = model
        self.adv_img = adv_img
        self.target = target
        self.eps = eps
        self.batch_size = batch_size
        self.ori_img = ori_img

    def seteps(self,eps):
        self.eps = eps

    def evaluate(self, x, pur_abs):  #这是适应度评估函数，这里传过来的x是pop，即初代小扰动。（pop_size*length个随机整数）,pur是相对于原始图像的初代大扰动，已经乘了pop_size份，和pop的shape一样
        x= x * self.eps#放大eps倍
        x_abs = np.abs(x)
        for i in range(len(x_abs)):
            x_abs[i] = np.clip(x_abs[i],0., pur_abs[i]) #将放大后的小扰动裁剪到大扰动的范围里
        x= x_abs*np.sign(x)# 换源小扰动的符号，保证生成的每个小扰动都和大扰动是逆方向
        with torch.no_grad():
            adv = np.clip(np.tile(self.adv_img, (len(x), 1, 1, 1)) + np.reshape(x, (len(x),) + self.adv_img.shape), 0., 1.)
            #注意这里这个adv_img是添加完大扰动后的原始图像了，adv就是添加完大扰动后再减去小扰动生成的对抗样本
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(adv, dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(adv, dtype=torch.float32))

        ''' np.tile：沿指定轴复制，这里是将image在第一维（张数）复制len（x）份，其他的长宽通道三个轴不变，于是便生成了pop（默认为50）个一样的image
        np.reshape（a,b）:是把a重塑成b的形式。这里要把原本的初始人口pop重塑成【(len(x),) + self.image.shape】，即image的shape，只不过第一维加上len（x）
        也就是说，重塑后的pop和tile后的image有一样的格式了，再乘上eps相加，就得出了这样一个tensor：50份原图像+50份不同的pop随机数。（即生成了50个对抗样本）
        然后再把生成得对抗样本裁剪到0-1，然后放到model里去predict'''
        ori_pred = np.asarray(predict)
        pred = ori_pred.copy()
        pred[pred >= (0.5 + 0)] = 1
        pred[pred < (0.5 + 0)] = -1
        # pred = np.argwhere(pred > 0)
        pop_target = np.tile(self.target, (len(x), 1))
        # print("pop_target",pop_target)
        # print("pred",pred)
        pred_no_match = []
        for i in range(len(pred)):
            if np.any(pred[i] != pop_target[i]):
                pred_no_match.append(i)
        # pred_no_match = np.argwhere((pred != pop_target))
        # print(pred_no_match)
        pur = (adv-np.clip(np.tile(self.ori_img, (len(x), 1, 1, 1)),0.,1.)) #(100, 3, 448, 448)，这里的pur是相对于原始图像的整体扰动
        pur = pur.reshape(pur.shape[0],-1)
        # print(pur.shape)
        fitness = np.linalg.norm(pur,axis=1, ord=2)
        # print(fitness.shape)
        # print(fitness)
        if(len(pred_no_match)!=0):
            # print("drop:",len(pred_no_match),", ",end=' ')
            fitness[pred_no_match] = 999
            # print(fitness)

        #np.newaxis增加一个维度，也就是把p中的每一个logits单独做成了一个list，比如[0.511],[0.205]
        return fitness



def mating4(pop,F):
    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    p4 = np.copy(p3)
    np.random.shuffle(p4)
    p5 = np.copy(p4)
    np.random.shuffle(p5)
    mutation = pop + F * (p2 - p3 + p4 - p5)
    return mutation

def mating(pop,F):

    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    mutation = pop + F * (p2 - p3)
    #off =  cross(pop, mutation,cr)
    return mutation

def mating_best(pop,fitness,F):
    best = np.argmin(fitness)  # best是当前最小fitness的编号
    mutation= np.copy(pop)
    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    for i in range(len(pop)):
        mutation[i] = pop[best] + F * (p2[i]-p3[i])
    return  mutation

def mating_mix(pop,fitness,fitmin,F):
    T = fitmin*2
    best = np.argmin(fitness) #best是当前最小fitness的编号
    p1 = np.arange(len(pop))  # p1,p2,p3是编号
    np.random.shuffle(p1)
    p2 = np.copy(p1)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    p4 = np.copy(p3)
    np.random.shuffle(p4)
    p5 = np.copy(p4)
    np.random.shuffle(p5)
    mutation = np.copy(pop)
    count = 0
    for i in range(len(pop)):
        prob = np.random.rand()
        if (prob < T):
            mutation[i] = pop[i] + F * (pop[p2[i]] - pop[p3[i]] )+0.25*( pop[p4[i]] - pop[p5[i]])
            count += 1
        else:
            mutation[i] = pop[i] + F * (pop[p2[i]] - pop[p3[i]])
    #print('Best率为' + str(1 - count/100) )
    return  mutation

def cross(pop, mutation,cr):
    M = np.random.random((len(pop), pop.shape[1])) <= cr  #上生成50行pop.shape[1]的浮点数，浮点数都是从0-1中随机。M是其中<=cr的下标
    #例如，cr=0.8，则说明交叉后有80%是来自变异后的子代，20%来自变异前的父代
    off_new = pop.copy()  #off_new选择所有当前pop
    off_new[M] = mutation[M].copy()  #off_new的m位换上mutation的m位
    return off_new #这就是新一代的子代

def select(pop,fitness,off,off_fitness):
   new_pop = pop.copy()
   new_fitness = fitness.copy()
   i=np.argwhere(fitness>off_fitness) #i是pop的fit大于子代fit的坐标
   # print("update:",i)
   new_pop[i] = off[i].copy() #对应位置换成子代
   new_fitness[i] = off_fitness[i].copy()
   return new_pop ,new_fitness

def meanpop(pop,sort,num):
    sum = pop[0]*0
    for j in range(num):
        i = sort[j]
        sum = sum+pop[i]
    mean = sum / num
    return mean


def test(r,model, image, target,eps):
    x_adv_tem = np.clip(image + np.reshape(r, image.shape) * eps, 0, 1)
    with torch.no_grad():
        if torch.cuda.is_available():
            adv_pred = model(torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32).cuda()).cpu()
        else:
            adv_pred = model(torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32))
    adv_pred = np.asarray(adv_pred)
    pred = adv_pred.copy()
    pred[pred >= (0.5 + 0)] = 1
    pred[pred < (0.5 + 0)] = -1
    #pred = np.argwhere(pred > 0)
    adv_pred_match_target = np.all((pred == target), axis=1)
    if adv_pred_match_target:
        return 1
    else: return 0



def IOU(put,box):
    s_box = (box[2] - box[0]) * (box[3] - box[1])
    s_put = ( put[2]-put[0])*(put[3]-put[1])
    s_overlap = (min(box[3],put[3])-max(box[1],put[1]))*(min(box[2],put[2])-max(box[0],put[0]))
    iou = max(0, s_overlap / (s_box + s_put - s_overlap))
    return iou

def GenBig( length, model, image, target,ori_loader,jump,yolonet,quit1,quit2,isblack):
    # return 0, False, 1,1
    img_dir = "../data/voc2007/VOCdevkit/VOC2007/JPEGImages/"
    imglist = os.listdir(img_dir)
    boxes = []
    target_label = np.argwhere(target > 0)  # [[1],[8]]
    # print(target_label)
    time = 0 #如果连续100次自己生成都失败的话，就说明该样本不适合粘贴
    giveup1 = quit1
    giveup2 = quit2
    if quit1 !=1:
        while time in range(200):
            x_adv = np.transpose(image, (1, 2, 0))
            if isblack==1:
                x_adv = x_adv*0
            x_adv = Image.fromarray(np.uint8(x_adv * 255))
            for num in range(len(target_label)):
                flag = target_label[num][0] #flag纪录本次循环中要贴的标签的下标
                file = open("../data/voclabels.txt")
                for i in range(flag):  # 将原标签拆分后，从原样本中顺着找满足单个标签的样本，然后扣下来贴在空白背景板上
                    skip = file.readline()
                lab = file.readline().split('\n')[0]  # lab是本次要找的标签的名称，比如“person“
                # print(lab)
                seed = np.random.uniform(0,9000)
                count = 0
                for img_name in imglist: #依次使用YOLO判断数据集每张图片包含的标签，并把符合攻击目标的部分裁剪下来贴在原样本上
                    if count<seed:
                        count+=1
                        continue
                    img = Image.open('../data/voc2007/VOCdevkit/VOC2007/JPEGImages/' + img_name)
                    windows=yolonet.detect_image(img)
                    # print(windows)
                    find = 0
                    for win in  windows:
                        if win[0] == lab:
                            # print(win[0])
                            pos = []
                            pos.append(win[1])
                            pos.append(win[2])
                            pos.append(win[3])
                            pos.append(win[4])
                            find=1
                            break
                    if find==1:
                        # print(pos)
                        part = img.crop(pos)
                        break
                for i in range(20):
                    x1 = int(np.random.uniform(448-part.size[0]))
                    y1 = int(np.random.uniform(448-part.size[1]))
                    put = [x1, y1,x1+part.size[0],y1+part.size[1]]
                    can = True
                    for box in boxes:
                        if (put[0]>box[0] and put[0]<box[2] ) or (put[1]>box[1] and put[1]<box[3]):
                            can = False
                    if can == True:
                        boxes.append(put)
                        break
                x_adv.paste(part, put)
            x_adv = np.asarray(x_adv)/255.
            x_adv = np.transpose(x_adv, (2, 0, 1))
            x_adv = np.clip(np.tile(x_adv, (1, 1, 1, 1)), 0., 1.)
            # 注意这里这个adv_img是添加完大扰动后的原始图像了，adv就是添加完大扰动后再减去小扰动生成的对抗样本
            with torch.no_grad():
                if torch.cuda.is_available():
                    predict = model(torch.tensor(x_adv, dtype=torch.float32).cuda()).cpu()
                else:
                    predict = model(torch.tensor(x_adv, dtype=torch.float32))
            ori_pred = np.asarray(predict)
            pred = ori_pred[0].copy()
            pred[pred >= (0.5 + 0)] = 1
            pred[pred < (0.5 + 0)] = -1
            adv_pred_match_target = np.all((pred == target))
            # target_label = np.argwhere(target > 0)
            # pred_label = np.argwhere(pred > 0)
            # print("target_label",target_label)
            # print("pred_label",pred_label)
            # x_adv = x_adv.squeeze()
            # x_tem = np.clip(x_adv, 0, 1)
            # x_tem = np.transpose(x_tem, (1, 2, 0))
            # x_tem = Image.fromarray(np.uint8(x_tem * 255))
            # plt.imshow(x_tem)
            # plt.axis('off')
            # plt.show()
            if adv_pred_match_target:
                x_adv = x_adv.squeeze()
                # x_tem = np.clip(x_adv, 0, 1)
                # x_tem = np.transpose(x_tem, (1, 2, 0))
                # x_tem = Image.fromarray(np.uint8(x_tem * 255))
                # plt.imshow(x_tem)
                # plt.axis('off')
                # plt.show()
                logging.info(("{}找到了").format(time))
                return x_adv ,True ,giveup1,giveup2
            else:
                time+=1
                # print(time,"次失败，再来！")
    giveup1=1 #从此以后直接跳过认识阶段
    if quit2 != 1:
        logging.info("难以生成，从数据集中粘贴")
        target_label = np.argwhere(target > 0) # [[1],[8]]
        # print(target_label)
        for time in range(1000):
            x_adv = np.transpose(image, (1, 2, 0))
            if isblack==1:
                x_adv = x_adv*0
            x_adv = Image.fromarray(np.uint8(x_adv * 255))
            boxes = []
            for num in range(len(target_label)):
                flag = target_label[num][0]  # flag纪录本次循环中要贴的标签的下标
                file = open("../data/voclabels.txt")
                for i in range(flag):  # 将原标签拆分后，从原样本中顺着找满足单个标签的样本，然后扣下来贴在空白背景板上
                    skip = file.readline()
                lab = file.readline().split('\n')[0]
                data = pd.read_csv("../data/voc2012/files/VOC2012/classification_mlliw_adv.csv")
                sum = 0
                for i in range(500):
                    if data.loc[i][flag + 1] == 1:
                        sum += 1
                count = 0
                seed = int(np.random.uniform(0, sum))
                for i in range(500):
                    if data.loc[i][flag + 1] == 1:
                        if count < seed:
                            count += 1
                            continue
                        img_name = ''
                        for j in range(6 - len(str(data.loc[i][0]))):
                            img_name += '0'
                        img_name += str(data.loc[i][0])
                        break
                dom = xml.dom.minidom.parse('../data/voc2007/VOCdevkit/VOC2007/Annotations/' + img_name + '.xml')
                root = dom.documentElement
                obj_list = root.getElementsByTagName('object')
                for obj in obj_list:
                    if obj.getElementsByTagName('name')[0].firstChild.data == lab:
                        pos = []
                        # print("Find  ",obj.getElementsByTagName('name')[0].firstChild.data)
                        pos.append(
                            int(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].firstChild.data))
                        pos.append(
                            int(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].firstChild.data))
                        pos.append(
                            int(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].firstChild.data))
                        pos.append(
                            int(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].firstChild.data))

                img = Image.open('../data/voc2007/VOCdevkit/VOC2007/JPEGImages/' + img_name + '.jpg')
                part = img.crop(pos)

                for i in range(20):
                    x1 = int(np.random.uniform(448 - part.size[0]))
                    y1 = int(np.random.uniform(448 - part.size[1]))
                    put = [x1, y1, x1 + part.size[0], y1 + part.size[1]]
                    can = True
                    for box in boxes:
                        if IOU(put, box) > 0.1:
                            can = False
                    if can == True:
                        boxes.append(put)
                        break
                x_adv.paste(part, put)
            x_adv = np.asarray(x_adv) / 255.
            x_adv = np.transpose(x_adv, (2, 0, 1))
            x_adv = np.clip(np.tile(x_adv, (1, 1, 1, 1)), 0., 1.)
            # 注意这里这个adv_img是添加完大扰动后的原始图像了，adv就是添加完大扰动后再减去小扰动生成的对抗样本
            with torch.no_grad():
                if torch.cuda.is_available():
                    predict = model(torch.tensor(x_adv, dtype=torch.float32).cuda()).cpu()
                else:
                    predict = model(torch.tensor(x_adv, dtype=torch.float32))
            ori_pred = np.asarray(predict)
            pred = ori_pred[0].copy()
            pred[pred >= (0.5 + 0)] = 1
            pred[pred < (0.5 + 0)] = -1
            adv_pred_match_target = np.all((pred == target))
            # x_adv = x_adv.squeeze()
            # x_tem = np.clip(x_adv, 0, 1)
            # x_tem = np.transpose(x_tem, (1, 2, 0))
            # x_tem = Image.fromarray(np.uint8(x_tem * 255))
            # plt.imshow(x_tem)
            # plt.axis('off')
            # plt.show()
            if adv_pred_match_target:
                x_adv = x_adv.squeeze()
                # x_tem = np.clip(x_adv, 0, 1)
                # x_tem = np.transpose(x_tem, (1, 2, 0))
                # x_tem = Image.fromarray(np.uint8(x_tem * 255))
                # plt.imshow(x_tem)
                # plt.axis('off')
                # plt.show()
                logging.info(("{}找到了").format(time))
                return x_adv, True,giveup1,giveup2
    giveup2=1
    logging.info("难以粘贴，从数据集中选")
    count = 0
    for i, (input, label) in enumerate(ori_loader):
        x_list = input[0].cpu().numpy()  # 这里的x_list是读进来一个batch的x, target是这一个batch的x的正确标签
        label = np.asarray(label)
        # print(input[0].shape) #[10, 3, 448, 448])
        # print(label.shape) #[10, 20]
        # print(x_list.shape) #(10, 3, 448, 448)
        for j in range(label.shape[0]):
            adv_pred_match_target = np.all((label[j] == target))
            target_label = np.argwhere(target > 0)
            ori_label = np.argwhere(label[j] > 0)
            if adv_pred_match_target:
                if count < jump:
                    count += 1
                    continue
                logging.info('跳过'+ str(count)+ '次，数据集中找到第'+str(i * label.shape[0] + j)+ "张")
                # print("**target", target_label)
                # print("**label", ori_label)
                return x_list[j], True,giveup1,giveup2
    return x_list[0], False,giveup1,giveup2


