#@Time      :2021/5/13 18:56
# @Author    :Klein
# @FileName  :ml_de.py
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
import random


class MLDE_hideall(object):
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
                i=0
                best_l2=999
                chance=0
                try_time=0
                while try_time <50:
                    diedai_tem = 0
                    target_label = np.argwhere(y_target[i] > 0)  # 返回的形式类似[[ 5] [14]]，就是第5，第14个为正
                    #print(x_list[i].shape)
                    istoobig = 1
                    giveup1 = 0 #是否跳过yolo拼接
                    giveup2 = 0 #是否跳过数据集拼接
                    while (istoobig == 1): #istoobig用来防止原始的扰动太大
                        # print("chance:",chance)
                        Adv, isAder,giveup1,giveup2 = GenBig(img_rows * img_cols * nchannels, self.model, x_list[i], y_target[i],ori_loader,try_time,yolonet,giveup1,giveup2)
                        if isAder == False:
                            print("重新生成对抗样本")
                            continue
                        if giveup2 ==1: #若不使用yolo拼贴和数据集拼贴，则设置取消初始扰动太大就重来的设定
                            chance=101
                        r, diedai_tem, l2 ,istoobig= DE(pop_size, generation, img_rows * img_cols * nchannels, self.model, x_list[i],
                                       target_label, eps, batch_size, y_target[i], Adv,chance)
                        chance+=1
                    x_adv_loop = np.clip(x_list[i] + np.reshape(r, x_list.shape[1:]) , 0, 1)
                    l2 = np.linalg.norm(x_adv_loop-x_list[i])
                    if l2 <best_l2:
                        best_l2 = l2
                        x_adv_tem = x_adv_loop
                    if l2 <= 40:
                        break
                    else:
                        logging.info("第"+str(try_time)+"次尝试失败，本次l2："+str(l2)+"当前最佳l2："+str(best_l2))
                        try_time+=1
                count += 1
                l2_sum += best_l2
                diedai += diedai_tem
                with torch.no_grad():
                    if torch.cuda.is_available():
                        adv_pred = self.model(
                            torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32).cuda()).cpu()
                    else:
                        adv_pred = self.model(torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32))
                adv_pred = np.asarray(adv_pred)
                pred = adv_pred.copy()
                pred[pred >= (0.5 + 0)] = 1
                pred[pred < (0.5 + 0)] = -1
                adv_pred_match_target = np.all((pred == y_target[i]), axis=1)
                if adv_pred_match_target:
                    success = success + 1
                    # x_tem = np.clip(x_list[i] , 0, 1)
                    # x_tem = np.transpose(x_tem, (1, 2, 0))
                    # x_tem = Image.fromarray(np.uint8(x_tem * 255))
                    # # plt.imshow(x_tem)
                    # # plt.axis('off')
                    # # plt.show()
                    # x_adv_tem = np.transpose(x_adv_tem, (1, 2, 0))
                    # x_adv_tem = Image.fromarray(np.uint8(x_adv_tem * 255))
                    # plt.imshow(x_adv_tem)
                    # plt.axis('off')
                    # plt.show()
                else:
                    fail.append(i)
                    logging.info('攻击失败的编号为：')
                    logging.info(fail)
                x_adv.append(x_adv_tem)
                #print("成功攻击"+str(success)+'/'+str(batch_size)+"个样本")
                logging.info('进度：' + str(count) + '/' + str(batch_size) + ',攻击成功' + str(success) + '个样本，当前成功率' + str(
                    success / count)+"该样本l2_norm:"+str(best_l2)+",平均l2："+str(l2_sum/success))
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

def DE(pop_size, generation, length, model, ori_image, target_label, eps, batch_size, target,Adv,chance):#Adv是一张目标类的样本或生成的符合目标的样本
    generation_save = np.zeros((10000,))
    curr_eps = eps
    problem = Problem(model, Adv, target, eps, batch_size, ori_image)  # 实例化一个problem对象
    pop = np.random.uniform(0, 1, size=(pop_size, length))  # 生成pop_size*length个随机整数，范围【0,1)；
    pur = Adv-ori_image #求扰动，方便后面算扰动的方向
    pur=pur.reshape(-1)#拍扁成1维
    sin_pur = pur
    sin_pur_abs = np.abs(pur)
    pur=pur[np.newaxis,:]
    pur=np.tile(pur, (pop_size, 1)) #增加一个维度并沿着该维度复制pop_size份，此时与pur的shape与pop一样了
    pur_abs = np.abs(pur)  #扰动的绝对值
    for i in range(len(pop)):
        pop[i]=np.random.uniform(0, pur_abs[i])
    pop = -1*pop*np.sign(pur)#保证生成的每个小扰动都和大扰动是逆方向
    # pop_size 是初始人口数量，默认50
    # length是指一幅图的img_rows * img_cols * nchannels，即展平后
    max_eval = pop_size * generation
    eval_count = 0
    fitness= problem.evaluate(pop,pur_abs)  # pop是初始扰动,返回的fitness是值，fit是未相加的值
    eval_count += pop_size
    count = 0
    fitmin = np.min(fitness)
    minl2=np.linalg.norm(Adv-ori_image)
    print("原始l2",minl2)
    generation_save[count] = fitmin
    last_fit = fitmin
    F = 0.5
    count_end=0
    count_rand=0
    randflag = 0
    best = np.argmin(fitness)
    x = pop[best] * curr_eps*0
    if minl2 >= 100 and chance < 100:
        print("原始l2太大，放弃了")
        return sin_pur + x, eval_count / 100, minl2, 1
    # fitness就是适应度函数。这里把适应度最小值放给generation_save[0]
     # np.where(fitness == 0)，返回满足条件的fitness的下标，若無，则继续
    while (eval_count < 200 *pop_size):  # 当进化代数小于最大代数max_eval
            count += 1
            if randflag == 1 :
                off = mating(pop, F)
            else:
                off = mating_best(pop,fitness,F)
            off_fitness  = problem.evaluate(off,pur_abs)  # 评估子代，返回子代的fitness:off_fitness_tem,另外还返回未sum的fit，用于update，尺寸为【pop,labels】
            fitmin = np.min(off_fitness)
            if fitmin == 999:
                if(curr_eps<=1):
                    print(str(count) + "次后,优化截止，当前最小l2范数：" + str(minl2))
                    break
                else:
                    curr_eps = curr_eps*0.8
                    problem.seteps(curr_eps)
                    print("减去的太大，调整eps为" +str(curr_eps))

                    pop = np.random.uniform(0, 1, size=(pop_size, length))  # 生成pop_size*length个随机整数，范围【0,1)；
                    pur = Adv - ori_image  # 求扰动，方便后面算扰动的方向
                    pur = pur.reshape(-1)  # 拍扁成1维
                    pur = pur[np.newaxis, :]
                    pur = np.tile(pur, (pop_size, 1))  # 增加一个维度并沿着该维度复制pop_size份，此时与pur的shape与pop一样了
                    pur_abs = np.abs(pur)  # 扰动的绝对值
                    for i in range(len(pop)):
                        pop[i] = np.random.uniform(0, pur_abs[i])
                    pop = -1 * pop * np.sign(pur)  # 保证生成的每个小扰动都和大扰动是逆方向4

                    fitness = problem.evaluate(pop, pur_abs)
                    continue
            eval_count += pop_size
            pop ,fitness = select (pop,fitness,off,off_fitness) #子代和父代相比，fitness小的留下来参与下一次迭代
            fitmin = np.min(fitness)
            if(fitmin<minl2 ):
                # print("最优结果已更新,", end='')
                best = np.argmin(fitness)
                x = pop[best] * curr_eps
                minl2 = fitmin
            # print(count,'curr fitmin:',fitmin,",min l2: ",minl2)
            generation_save[count] = fitmin  # 再记录fitnss的最小值
            if (count == 10 and fitmin >= 100):
                print("10次之内没降下来，初始化失败")
                break
            if (fitmin <= 1):
                print("扰动足够小，提前终止")
                break
            if (fitmin == last_fit):
                count_end += 1
                if (count_end == 11):
                    # if(curr_eps<=0.5):
                    print("超过10次迭代不下降，失败")
                    break
            else:
                last_fit = fitmin
                count_end = 0
            if (curr_eps < 0.5):
                print("eps太小，失败")
                break
            if(last_fit-fitmin)<0.5 :
                count_rand+=1
                if (count_rand == 10):
                    if fitmin > minl2:
                        print("没必要再调低了，结束" )
                        break
                    curr_eps = curr_eps * 0.8
                    problem.seteps(curr_eps)
                    print("连续10次变化不明显，调整eps为" + str(curr_eps))
                    count_rand = 0
                    pop = np.random.uniform(0, 1, size=(pop_size, length))  # 生成pop_size*length个随机整数，范围【0,1)；
                    pur = Adv - ori_image  # 求扰动，方便后面算扰动的方向
                    pur = pur.reshape(-1)  # 拍扁成1维
                    pur = pur[np.newaxis, :]
                    pur = np.tile(pur, (pop_size, 1))  # 增加一个维度并沿着该维度复制pop_size份，此时与pur的shape与pop一样了
                    pur_abs = np.abs(pur)  # 扰动的绝对值
                    for i in range(len(pop)):
                        pop[i] = np.random.uniform(0, pur_abs[i])
                    pop = -1 * pop * np.sign(pur)  # 保证生成的每个小扰动都和大扰动是逆方向

                    fitness = problem.evaluate(pop, pur_abs)
                    fitmin = np.min(fitness)
            else:
                count_rand=0

    x_abs = np.abs(x)
    for i in range(len(x_abs)):
        x_abs[i] = np.clip(x_abs[i], 0., sin_pur_abs[i])  # 将放大后的小扰动裁剪到大扰动的范围里
    x = x_abs * np.sign(x)  # 换源小扰动的符号，保证生成的每个小扰动都和大扰动是逆方向
    return sin_pur+x, eval_count/100, minl2, 0



def IOU(put,box):
    s_box = (box[2] - box[0]) * (box[3] - box[1])
    s_put = ( put[2]-put[0])*(put[3]-put[1])
    s_overlap = (min(box[3],put[3])-max(box[1],put[1]))*(min(box[2],put[2])-max(box[0],put[0]))
    iou = max(0, s_overlap / (s_box + s_put - s_overlap))
    return iou

def rndColor():
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

def GenBig( length, model, image, target,ori_loader,jump,yolonet,quit1,quit2):
    img_dir = "../data/voc2007/VOCdevkit/VOC2007/JPEGImages/"
    imglist = os.listdir(img_dir)
    boxes = []
    target_label = np.argwhere(target > 0)  # [[1],[8]]
    time = 0 #如果连续100次自己生成都失败的话，就说明该样本不适合粘贴
    giveup1 = 0
    giveup2 = 0
    while time in range(100):
        x_adv = np.transpose(image, (1, 2, 0))
        x_adv = Image.fromarray(np.uint8(x_adv * 255))
        boxes=yolonet.detect_image(x_adv)
        for box in boxes :
            # print(box)
            draw = ImageDraw.Draw(x_adv)
            draw.rectangle(((box[1], box[2]),(box[3], box[4])), fill=rndColor() )
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
        if adv_pred_match_target:
            x_adv = x_adv.squeeze()
            x_tem = np.clip(x_adv, 0, 1)
            x_tem = np.transpose(x_tem, (1, 2, 0))
            x_tem = Image.fromarray(np.uint8(x_tem * 255))
            plt.imshow(x_tem)
            plt.axis('off')
            plt.show()
            print(time,"找到了")
            return x_adv ,True ,giveup1,giveup2
        else:
            time+=1
            x_adv = x_adv.squeeze()
            x_tem = np.clip(x_adv, 0, 1)
            x_tem = np.transpose(x_tem, (1, 2, 0))
            x_tem = Image.fromarray(np.uint8(x_tem * 255))
            plt.imshow(x_tem)
            plt.axis('off')
            plt.show()
            print(np.argwhere(pred > 0))
            print(time,"次失败，再来！")

    return 0, False,giveup1,giveup2


