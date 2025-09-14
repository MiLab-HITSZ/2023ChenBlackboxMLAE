"""
chenzhijian

@author: chenzhijian
"""

import os
import json
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
print(torch.cuda.is_available())
print(torch.__version__)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.fft
import torchvision.transforms as transforms
import numpy as np
import argparse
# from tqdm import tqdm
import time
import requests
import torchvision

from PIL import Image
from torchvision import transforms as T
from torchvision.io import read_image


from ml_gcn_model.models import gcn_resnet101_attack
from ml_gcn_model.voc import Voc2007Classification
from ml_gcn_model.util import Warp
from ml_gcn_model.voc import write_object_labels_csv
from ml_liw_model.models import inceptionv3_attack
from ml_liw_model.voc import Voc2012Classification
from ml_gcn_model.util import Warp
from ml_liw_model.voc import write_object_labels_csv


# from demo_voc2007_gcn import  *

import torch.nn as nn
import torchvision.datasets as dsets

import torchvision.transforms as transforms
import torchvision.models as torch_models
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from utils import get_label
from utils import valid_bounds, clip_image_values
from PIL import Image
from torch.autograd import Variable
from numpy import linalg 
import foolbox
import math
from generate_2d_dct_basis import *
import time

###############################################################
###############################################################
# Parameters
grad_estimator_batch_size = 4    # batch size for GeoDA
verbose_control = 'Yes'
# verbose_control = 'No'
Q_max = 300
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
sub_dim=75
tol = 0.0001
sigma = 0.0002
mu = 0.6

dist = 'l2'
# dist = 'linf'

# search_space = None
search_space = 'sub'

###########        all    ################
ori_folder = './cocoori/'
ori_448_folder = './cocoori448/'
if not os.path.exists(ori_448_folder):
    os.makedirs(ori_448_folder)
adv_save = './adv_save/'
attacktype = 'hideall'
mooodel = 'liwcoco'
netmodel = 'liwcoco'

###########     single       #################
# mooodel = 'liw2007'
# ori_folder = './hidesingle/ori_{}/'.format(mooodel)
# ori_448_folder = './hidesingle/ori_{}_448/'.format(mooodel)
# if not os.path.exists(ori_448_folder):
#     os.makedirs(ori_448_folder)
# adv_save = './adv_save/'
# attacktype = 'hidesingle'
# ini_path = './hidesingle/adv_{}/'.format(mooodel)

#############################
#   'gcn2012'  'gcn2007' 'liw2007'  'liw2012'

image_iter = 0
##################hide all ###################
ori_imge = os.listdir(ori_folder)
# #voc
# ori_imge.sort(key=lambda x: int(x[-7:-4]))
# #coco
ori_imge.sort(key=lambda x: int(x[-10:-4]))
#nuswide
# ori_imge.sort(key=lambda x: int(x[0:6]))

###############################################################
# Functions
###############################################################
def get_model(x):
    num_classes = 20
    # load torch model
    if x == 'gcn2007':
        model = gcn_resnet101_attack(num_classes=num_classes,
                                     t=0.4,
                                     adj_file='./data/voc2007/voc_adj.pkl',
                                     word_vec_file='./data/voc2007/voc_glove_word2vec.pkl',
                                     save_model_path='./checkpoint/mlgcn/voc2007/voc_checkpoint.pth.tar')
    elif x == 'gcn2012':
        model = gcn_resnet101_attack(num_classes=num_classes,
                                     t=0.4,
                                     adj_file='./data/voc2012/voc_adj.pkl',
                                     word_vec_file='./data/voc2012/voc_glove_word2vec.pkl',
                                     save_model_path='./checkpoint/mlgcn/voc2012/voc_checkpoint.pth.tar')
    elif x == 'gcncoco':
        model = gcn_resnet101_attack(num_classes=80,
                                     t=0.4,
                                     adj_file='./data/coco/coco_adj.pkl',
                                     word_vec_file='./data/coco/coco_glove_word2vec.pkl',
                                     save_model_path='./checkpoint/mlgcn/coco/coco_checkpoint.pth.tar')
    elif x == 'gcnnus':
        model = gcn_resnet101_attack(num_classes=81,
                                     t=0.4,
                                     adj_file='./data/NUSWIDE/nuswide_adj.pkl',
                                     word_vec_file='./data/NUSWIDE/glove_word2vec.pkl',
                                     save_model_path='./checkpoint/mlgcn/NUSWIDE/model_best.pth.tar')
    elif x == 'liwcoco':
        model = inceptionv3_attack(num_classes=80,
                                   save_model_path='./checkpoint/mlliw/cocon/model_best.pth.tar')

    elif x == 'liwnus':
        model = inceptionv3_attack(num_classes=81,
                                   save_model_path='./checkpoint/mlliw/NUSWIDE/model_best.pth.tar')
    elif x == 'liw2007':
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2007/model_best.pth.tar')

    elif x == 'liw2012':
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2012/model_best.pth.tar')

    return model

###############################################################
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def inv_tf(x, mean, std):

    for i in range(len(mean)):
        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)

    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)

    return x

###############################################################

def from_np_to_01(imnp):
    im01 = inv_tf(imnp.cpu().numpy().squeeze(), mean, std)
    im01= np.transpose(im01, (2, 0, 1))
    return im01

###############################################################
def topk_3D (grad, k):


    grad_flatten = grad.cpu().numpy().reshape(-1)
    grad_flatten_torch = torch.tensor(grad_flatten)
    topk, indices = torch.topk(torch.abs(grad_flatten_torch), k)
    #grad_k_flatten = torch.zeros([2*4*3])

    grad_k_flatten = torch.zeros([448*448*3])

    for ind in indices:

        grad_k_flatten[ind] = grad_flatten[ind] + 0

    grad_k_flatten_np = grad_k_flatten.cpu().numpy()

    #grad_k_3D_np = np.reshape(grad_k_flatten_np, ( 3, 2, 4))

    grad_k_3D_np = np.reshape(grad_k_flatten_np, ( 3, 448, 448))

    grad_3D_torch = torch.tensor(grad_k_3D_np)
    grad_3D_sign = torch.sign(grad_3D_torch)


    return grad_3D_sign

###############################################################


def is_adversarial(given_image, tar_label):

    predict_label =(net(given_image) >= 0.5) + 0
    pre = np.asarray(predict_label.cpu())
    return (pre == tar_label).all()

###############################################################

def find_random_adversarial(image, epsilon=1000):

    num_calls = 1

    alp = 0.3
    perturbed = image

    while is_adversarial(perturbed, tar_label) == 0 and alp<=1.0:
        # print(is_adversarial(perturbed, tar_label))


        random_matrix = torch.rand_like(perturbed)

        random_matrix = random_matrix.to(device)

        perturbed = alp*random_matrix + (1-alp)* image
        alp += 0.05

        perturbed = torch.clip(perturbed, 0, 1)
        perturbed = perturbed.to(device)
        num_calls += 1


    return perturbed, num_calls

###############################################################

def bin_search(x_0, x_random, tol):

    num_calls = 0
    adv = x_random
    cln = x_0
    low = 0.
    high = 1.

    n_step = 0
    while True:
        mid = (cln + adv) / 2.0
        num_calls += 1

        if is_adversarial(mid, tar_label):
            adv = mid
        else:
            cln = mid

        if torch.norm(adv - cln).cpu().numpy() < tol:

            if is_adversarial(adv, tar_label):
                print('Bi success!!!')

            break
    return adv, num_calls, is_adversarial(adv, tar_label)

###############################################################

def black_grad_batch(x_boundary, q_max, sigma, random_noises, batch_size, tar_label):

    grad_tmp = [] # estimated gradients in each estimate_batch
    z = []        # sign of grad_tmp
    outs = []
    num_batchs = math.ceil(q_max/batch_size)
    last_batch = q_max - (num_batchs-1)*batch_size

    EstNoise = SubNoise(batch_size, sub_basis_torch).cuda()
    all_noises = []
    for j in range(num_batchs):
        if j == num_batchs-1:
            EstNoise_last = SubNoise(last_batch, sub_basis_torch).cuda()
            current_batch = EstNoise_last()
            current_batch_np = current_batch.cpu().numpy()
            noisy_boundary = [x_boundary[0,:,:,:].cpu().numpy()]*last_batch +sigma*current_batch.cpu().numpy()


        else:
            current_batch = EstNoise()
            current_batch_np = current_batch.cpu().numpy()
            noisy_boundary = [x_boundary[0,:,:,:].cpu().numpy()]*batch_size +sigma*current_batch.cpu().numpy()

        all_noises.append(current_batch_np)

        noisy_boundary_tensor = torch.tensor(noisy_boundary).to(device)

        # predict_labels = torch.argmax(net.forward(noisy_boundary_tensor),1).cpu().numpy().astype(int)

        predict_labels = (net(noisy_boundary_tensor) >= 0.5) + 0
        predict_labels11 = np.asarray(predict_labels.cpu())

        outs.append(predict_labels11)
    all_noise = np.concatenate(all_noises, axis=0)
    outs = np.concatenate(outs, axis=0)

    for i, predict_label in enumerate(outs):
        if (predict_label !=tar_label).any():
            z.append(1)
            grad_tmp.append(all_noise[i])
        else:
            z.append(-1)
            grad_tmp.append(-all_noise[i])

    grad = -(1/q_max)*sum(grad_tmp)


    grad_f = torch.tensor(grad).to(device)[None, :,:,:]

    return grad_f, sum(z)

###############################################################

def go_to_boundary(x_0, grad, x_b):

    epsilon = 0.05
    num_calls = 1
    perturbed = x_0

    if dist == 'l1' or dist == 'l2':
        grads = grad
    if dist == 'linf':
        grads = torch.sign(grad)/torch.norm(grad)

    while is_adversarial(perturbed, tar_label) == 0:
        # print(torch.max(num_calls*epsilon* grads[0]))
        perturbed = x_0 + (num_calls*epsilon* grads[0])
        perturbed = torch.clip(perturbed, 0, 1)
        num_calls += 1
        if num_calls > 200:
            break
    if num_calls<=200:
        print('Go to Boundary Success!!!')
        return perturbed, num_calls, epsilon * num_calls
    else:
        # print(print('Neer Boundary!!!'))
        return perturbed, num_calls, epsilon * num_calls


###############################################################
def GeoDA(x_b, iteration, q_opt):

    norms = []
    q_num = 0
    grad = 0
    for i in range(iteration):
        t1 = time.time()
        random_vec_o = torch.randn(q_opt[i],3,448,448)
        grad_oi, ratios = black_grad_batch(x_b, q_opt[i], sigma, random_vec_o, grad_estimator_batch_size , tar_label)
        q_num = q_num + q_opt[i]
        grad = grad_oi + grad

        x_adv, qs, eps = go_to_boundary(x_0, grad, x_b)
        if qs >200:
            print('Fail, break!!!')
            return x_b, q_num, grad

        q_num = q_num + qs
        x_adv11, bin_query,if_update = bin_search(x_0, x_adv, tol)

        if not if_update:
            break
        else:
            x_adv = x_adv11

        q_num = q_num + bin_query

        x_b = x_adv

        t2 = time.time()
        if dist == 'l1' or dist == 'l2':
            dp = 'l2'
            norm_p = linalg.norm((x_adv-im_ini).cpu().numpy().squeeze())
        if dist == 'linf':
            dp = dist
            norm_p = np.max(abs((x_adv-im_ini).cpu().numpy().squeeze()))

        if verbose_control == 'Yes':
            message = ' (took {:.5f} seconds)'.format(t2 - t1)
            print('iteration -> ' + str(i) + str(message) + '     -- ' + dp + ' norm is -> ' + str(norm_p))


    x_adv = torch.clip(x_adv, 0, 1)

    predict_x_adv = (net(x_adv) >= 0.5) + 0
    prepredict_x_adv = np.asarray(predict_x_adv.cpu())
    print(prepredict_x_adv)
    print('x_adv  is adv',(prepredict_x_adv == tar_label).all())
    return x_adv, q_num, grad

###############################################################

def opt_query_iteration(Nq, T, eta):
    coefs=[eta**(-2*i/3) for i in range(0,T)]
    coefs[0] = 1*coefs[0]
    sum_coefs = sum(coefs)
    opt_q=[round(Nq*coefs[i]/sum_coefs) for i in range(0,T)]
    if opt_q[0]>80:
        T = T + 1
        opt_q, T = opt_query_iteration(Nq, T, eta)
    elif opt_q[0]<50:
        T = T - 1

        opt_q, T = opt_query_iteration(Nq, T, eta)

    return opt_q, T

def uni_query(Nq, T, eta):

    opt_q=[round(Nq/T) for i in range(0,T)]


    return opt_q

###############################################################

def load_image(image, shape=(448, 448), data_format='channels_last'):

    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']

    image = image.resize(shape)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    return image

###############################################################

class SubNoise(nn.Module):
    """given subspace x and the number of noises, generate sub noises"""
    # x is the subspace basis
    def __init__(self, num_noises, x):
        self.num_noises = num_noises
        self.x = x
        super(SubNoise, self).__init__()
        # print('x.shape',x.shape)
        # print('type(x)',type(x))

    def forward(self):
        ############### 第一个是不用dct##############
        r = torch.zeros([448 ** 2, 3*self.num_noises], dtype=torch.float32)

        ####### 这里是用DCT    #################
        noise = torch.randn([self.x.shape[1], 3*self.num_noises], dtype=torch.float32).cuda()
        # print(self.x.shape)
        # print(noise.shape)
        sub_noise = torch.transpose(torch.mm(self.x, noise), 0, 1)
        r = sub_noise.view([ self.num_noises, 3, 448, 448])
        r_list = r
        # print('r_list',r_list.shape)
        # print('r_list type',type(r_list))
        return r_list
###############################################################
if search_space == 'sub':
    # print('Check if DCT basis available ...')
    path = os.path.join(os.path.dirname(__file__), '2d_dct_basis_{}.npy'.format(sub_dim))
    if os.path.exists(path):
        # print('Yes, we already have it ...')
        sub_basis = np.load('2d_dct_basis_{}.npy'.format(sub_dim)).astype(np.float32)
    else:
        print('Generating dct basis ......')
        sub_basis = generate_2d_dct_basis(sub_dim).astype(np.float32)
        print('Done!\n')

    estimate_batch = grad_estimator_batch_size
    sub_basis_torch = torch.from_numpy(sub_basis).cuda()

    EstNoise = SubNoise(estimate_batch, sub_basis_torch).cuda()
    random_vectors = EstNoise()
    random_vectors_np = random_vectors.cpu().numpy()



#############        ATTACK          ###############################
#####
# Models

print('ATTACK BEGIN.........\n')

print('model load.........\n')
resnet50 = get_model(mooodel).eval()
if torch.cuda.is_available():
    resnet50 = resnet50.cuda()

meanfb = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
stdfb = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


fmodel = foolbox.models.PyTorchModel(
    resnet50, bounds=(0, 1), num_classes=20, preprocessing=(meanfb, stdfb))

# Check for cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a pretrained model
net = get_model(netmodel).eval()
if torch.cuda.is_available():
    net = net.cuda()

###################################

##
## Load Image and Resize

adv_save_path = os.path.join(adv_save, mooodel,attacktype,'{}'.format(Q_max))
if not os.path.exists(adv_save_path):
    os.makedirs(adv_save_path)
print(adv_save_path)


##################################single ###########################

# ori_imge = os.listdir(ori_folder)
# ori_imge.sort(key=lambda x: int(x[0:-8]))
#
# adv_save_path = os.path.join(adv_save, mooodel,attacktype,'{}'.format(Q_max))
# if not os.path.exists(adv_save_path):
#     os.makedirs(adv_save_path)
# print(adv_save_path)
#
# ini_imge = os.listdir(ini_path)
# ini_imge.sort(key=lambda x: int(x[0:-8]))


numi = 0
# tt =[]
with torch.no_grad():
    for img in ori_imge:
        numi+=1
        # if numi>25:
        #     break

        print('\nThis is num {} PIC'.format(numi))
        print(img)

        t11 = time.time()
        im_orig = Image.open(os.path.join(ori_folder, img)).resize((448, 448))

        im_orig_np = np.array(im_orig).transpose(2,0,1)/255.
        # print(im_orig_np.shape)

        im_orig.save(ori_448_folder + "{}_adv.png".format(numi))


        im_ini = torch.tensor(im_orig_np).float().unsqueeze(0).cuda()



#################   single ############
        # iniadv = Image.open(os.path.join(ini_path, img)).resize((448, 448))
        # iniadv_np = np.array(iniadv).transpose(2, 0, 1) / 255.
        # ini_adv = torch.tensor(iniadv_np).float().unsqueeze(0).cuda()
###################

        x_0 = torch.tensor(im_orig_np).float().unsqueeze(0).cuda()
        # print(torch.max(x_0),torch.min(x_0))

        pred = (net(x_0) >= 0.5) + 0
        now_label = np.where(np.asarray(pred.cpu())>0)

        print('ori label',pred,now_label[1])

#         tt.append(now_label[1])
#
# print(tt)




#
        x_0_np = x_0.cpu().numpy()

        now_label = np.asarray(pred.cpu())
        # print('clean img labels:',now_label)

       ######################## single
        # tar_label = np.asarray(((net(ini_adv) >= 0.5) + 0).cpu())

        tar_label = np.zeros_like(pred.cpu())
#############################


        if (now_label == tar_label).all():
            print('Already missclassified ... Lets try another one!')
            im_orig.save(adv_save_path + "/{}_adv.png".format(numi))


        else:
            image_iter = image_iter + 1

        ###################################

            print('generate ini adv  with random')
            x_random, query_random_1 = find_random_adversarial(x_0, epsilon=100)

            ################### single#############
            # print('generate ini adv  with ini')
            # x_random = ini_adv
            # query_random_1 = 1



            print('Ini adv :',is_adversarial(x_random, tar_label))

            # print('Binary search \n')
            # Binary search
            x_boundary, query_binsearch_2, if_update = bin_search(x_0, x_random, tol)

            if if_update:

                # print('Binary search end\n')
                print('Bi result is adv:',is_adversarial(x_boundary, tar_label))

                x_b = x_boundary

                query_rnd = query_binsearch_2 + query_random_1
                ###################################
                # Run over iterations
                iteration = round(Q_max/300)
                q_opt_it = int(Q_max  - (iteration)*25)
                q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, mu )

                print('#################################################################')
                print('Start: The GeoDA will be run for:' + ' Iterations = ' + str(iterate) + ', Query = ' + str(Q_max) + ', Norm = ' + str(dist)+ ', Space = ' + str(search_space) )
                print('#################################################################')

                t3 = time.time()
                x_adv, query_o, gradient = GeoDA(x_b, iterate, q_opt_iter)

                x_adv11222,_ ,if_update= bin_search(x_0,x_adv,tol)

                if if_update:
                    x_adv = x_adv11222


                t4 = time.time()
                message = ' took {:.5f} seconds'.format(t4 - t3)
                qmessage = ' with query = ' + str(query_o + query_rnd)

                print('Final adv is adv ?',is_adversarial(x_adv, tar_label))

                pic = x_adv.cpu().numpy().squeeze()
                pic = np.transpose(pic, (1, 2, 0))
                print(pic.shape)
                imgggg = Image.fromarray(np.uint8(pic * 255.))
                imgggg.save(adv_save_path + "/{}_adv.png".format(numi))

                print('#################################################################')
                print('End: The GeoDA algorithm' + message + qmessage )
                print('#################################################################')

            else:
                picf = x_random.cpu().numpy().squeeze()
                picf = np.transpose(picf, (1, 2, 0))
                impicf = Image.fromarray(np.uint8(picf * 255.))
                impicf.save(adv_save_path + "/{}_adv.png".format(numi))






