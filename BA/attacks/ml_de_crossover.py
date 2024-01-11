#@Time      :2021/5/13 18:56
# @Author    :Klein
# @FileName  :ml_de.py
import numpy as np
import logging
import torch
import gc
from multiprocessing import Pool
from ml_liw_model.train  import criterion
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
class MLDE(object):
    def __init__(self, model):
        self.model = model

    def generate_np(self, x_list, **kwargs):
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
        use_grad = kwargs['use_grad']
        x_adv = [] #一张一张保存生成的对抗样本
        success = 0
        nchannels,img_rows, img_cols,  = x_list.shape[1:4]
        if use_grad:
            logging.info('使用梯度')
            x_t = torch.tensor(x_list)
            y_target_t = torch.tensor(y_target)
            x_t.requires_grad = True
            if torch.cuda.is_available():
                x_t = x_t.cuda()
                y_target_t = y_target_t.cuda()
            output = self.model(x_t)
            loss = criterion(output, y_target_t)
            gradient = -1*torch.autograd.grad(loss, x_t, create_graph=True)[0]
            gradient = gradient.detach().cpu().numpy()
        count = 0
        diedai = 0
        fail = []
        #np.random.seed(1)
        for i in range(len(x_list)):
                diedai_tem = 0
                target_label = np.argwhere(y_target[i] > 0)  # 返回的形式类似[[ 5] [14]]，就是第5，第14个为正
                if use_grad:
                    r, diedai_tem, _2 = DE(pop_size, generation, img_rows * img_cols * nchannels, self.model, x_list[i],
                                   target_label, eps, batch_size, gradient[i])
                else:
                    r, diedai_tem, _2 = DE(pop_size, generation, img_rows * img_cols * nchannels, self.model, x_list[i],
                                   target_label, eps, batch_size, gradient=None)
                x_adv_tem = np.clip(x_list[i] + np.reshape(r, x_list.shape[1:]) * eps, 0, 1)
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
                    '''x_tem = np.clip(x_list[i] , 0, 1)
                    x_tem = np.transpose(x_tem, (1, 2, 0))
                    x_tem = Image.fromarray(np.uint8(x_tem * 255))
                    plt.imshow(x_tem)
                    plt.axis('off')
                   # plt.show()
                    #x_tem.save('../IMG/2007/'+str(i)+'0.jpg')
                    x_adv_tem = np.transpose(x_adv_tem, (1, 2, 0))
                    x_adv_tem = Image.fromarray(np.uint8(x_adv_tem * 255))
                    plt.imshow(x_adv_tem)
                    plt.axis('off')
                    #plt.show()
                    x_adv_tem.save('../IMG/2007/' + str(i) + '2.jpg')'''
                else:
                    fail.append(i)
                    logging.info('攻击失败的编号为：')
                    logging.info(fail)
                x_adv.append(x_adv_tem)
                #print("成功攻击"+str(success)+'/'+str(batch_size)+"个样本")
                count+=1
                logging.info('进度：' + str(count) + '/' + str(batch_size) + ',攻击成功' + str(success) + '个样本，当前成功率' + str(
                    success / count)+',该样本迭代'+str(diedai_tem)+'次，平均迭代'+str(diedai/count))
        return x_adv , diedai

class Problem:
    def __init__(self, model, image, target_label, eps, batch_size):
        self.model = model
        self.image = image
        self.target_label = target_label
        self.eps = eps
        self.batch_size = batch_size


    def evaluate(self, x):  #这是适应度评估函数，这里传过来的x是pop，即初代扰动。（pop_size*length个随机整数）
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.image, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.image.shape) * self.eps, 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.image, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.image.shape) * self.eps, 0.,
                                                          1.), dtype=torch.float32))

        ''' np.tile：沿指定轴复制，这里是将image在第一维（张数）复制len（x）份，其他的长宽通道三个轴不变，于是便生成了pop（默认为50）个一样的image
        np.reshape（a,b）:是把a重塑成b的形式。这里要把原本的初始人口pop重塑成【(len(x),) + self.image.shape】，即image的shape，只不过第一维加上len（x）
        也就是说，重塑后的pop和tile后的image有一样的格式了，再乘上eps相加，就得出了这样一个tensor：50份原图像+50份不同的pop随机数。（即生成了50个对抗样本）
        然后再把生成得对抗样本裁剪到0-1，然后放到model里去predict'''

        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        # Unconstrained optimization
        fitness = fitness[:, np.newaxis]
        #np.newaxis增加一个维度，也就是把p中的每一个logits单独做成了一个list，比如[0.511],[0.205]
        return fitness, fit

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

def mating_best(pop,fitness,F,):
    p1 = np.arange(len(pop)) #p1,p2,p3是编号
    p2 = np.copy(p1)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    mutation = np.copy(pop)
    for i in range(len(pop)):
        if(fitness[p1[i]]>fitness[p2[i]]  and fitness[p1[i]]>fitness[p3[i]]):
            mutation[i] = pop[p1[i]]+F*(pop[p2[i]]-pop[p3[i]])
        elif(fitness[p2[i]]>fitness[p1[i]]  and fitness[p2[i]]>fitness[p3[i]]):
            mutation[i] = pop[p2[i]]+F*(pop[p1[i]]-pop[p3[i]])
        else:
            mutation[i] = pop[p3[i]] + F * (pop[p1[i]] - pop[p2[i]])
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
    M = np.random.random((len(pop), pop.shape[1])) <= cr  #上生成50行pop.shape[1]的浮点数，浮点数都是从0-1中随机。M是其中<=0.5的下标
    off_new = pop.copy()  #off_new选择所有当前pop
    off_new[M] = mutation[M].copy()  #off_new的m位换上mutation的m位
    return off_new #这就是新一代的子代

def select(pop,fitness,fit,off,off_fitness,off_fit):
   new_pop = pop.copy()
   new_fitness = fitness.copy()
   new_fit = fit.copy()
   i=np.argwhere(fitness>off_fitness) #i是pop的fit大于子代fit的坐标
   new_pop[i] = off[i].copy() #对应位置换成子代
   new_fitness[i] = off_fitness[i].copy()
   new_fit[i] = off_fit[i].copy()
   return new_pop ,new_fitness ,new_fit

def meanpop(pop,sort,num):
    sum = pop[0]*0
    for j in range(num):
        i = sort[j]
        sum = sum+pop[i]
    mean = sum / num
    return mean

def update (fit,pop, fitness,problem) :
    #print('新一次update开始')
    popnew = pop.copy()
    sort = np.argsort(fitness.reshape(-1)) #排序之前要把fitness展成1维。原版本是【[48],[47],[50]】，现在是【48,47,50】
    for q in range (len(pop)):
        i = sort[q]  # i是fitness最小的下标
        fit_item = fit.copy()
        c = np.argwhere(fit[i] == 0)
        fit_item[:, c] = 0
        fitness_tem = np.sum(fit_item, axis=1)
        j = np.argmin(fitness_tem)
        popnew[i] = pop[i] + pop[j]*0.5
    off_fitness_new, off_fit_new = problem.evaluate(popnew)
    pop1, fitness1, fit1 = select(pop, fitness, fit, popnew, off_fitness_new, off_fit_new)  # update之前和之后相比，fitness小的留下来参与下一次迭代
    return pop1,fitness1, fit1


def DE(pop_size, generation, length, model, image, target_label, eps, batch_size, gradient):
    generation_save = np.zeros((10000,))
    problem = Problem(model, image, target_label, eps, batch_size)  # 实例化一个problem对象
    pop = np.random.uniform(-1, 1, size=(pop_size, length))  # 生成pop_size*length个随机整数，范围【-1,2),也就是-1，0，1；
    # pop_size 是初始人口数量，默认50
    # length是指一幅图的img_rows * img_cols * nchannels，即展平后
    if (not (gradient is None)):  # gradient是该图片的梯度信息
        pop[0] = np.reshape(np.sign(gradient), (length))  # 即：若需要梯度信息，50个初始人口的第一个，使用展平的梯度，而不是随机的-1，0，1值
    max_eval = pop_size * generation
    eval_count = 0
    fitness, fit = problem.evaluate(pop)  # pop是初始扰动,返回的fitness是值，fit是未相加的值
    eval_count += pop_size
    count = 0
    fitmin = np.min(fitness)
    fitcount = 0
    generation_save[count] = fitmin
    fitT = fitmin
    F = 0.5
    count5=0
    # fitness就是适应度函数。这里把适应度最小值放给generation_save[0]
    if (len(np.where(fitness == 0)[0]) == 0):  # np.where(fitness == 0)，返回满足条件的fitness的下标，若無，则继续
        while (eval_count < max_eval):  # 当进化代数小于最大代数
            count += 1
            off = mating(pop,F)
            #off = cross(pop, mutation,0.8)
            off_fitness , off_fit = problem.evaluate(off)  # 评估子代，返回子代的fitness:off_fitness_tem,另外还返回未sum的fit，用于update，尺寸为【pop,labels】
            eval_count += pop_size
            pop ,fitness ,fit = select (pop,fitness,fit,off,off_fitness,off_fit) #子代和父代相比，fitness小的留下来参与下一次迭代
            fitmin = np.min(fitness)
            generation_save[count] = fitmin  # 再记录fitnss的最小值
            #print('第'+str(count)+'次迭代，fitness最小值：'+str(generation_save[count]))
            if (len(np.where(fitness == 0)[0]) != 0):
                break
    if (len(np.where(fitness == 0)[0]) != 0):
        return pop[np.where(fitness == 0)[0][0]], eval_count/100, generation_save[:count + 1]
    else:
        return pop[0], eval_count/100, generation_save[:count + 1]
