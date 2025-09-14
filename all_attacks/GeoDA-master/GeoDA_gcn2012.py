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
grad_estimator_batch_size =2    # batch size for GeoDA
verbose_control = 'Yes'
# verbose_control = 'No'
Q_max = 300
torch.manual_seed(998)
torch.cuda.manual_seed(998)
np.random.seed(998)
sub_dim=75
tol = 0.01
sigma = 0.0002
mu = 0.6

#dist = 'l2'
dist = 'linf'
dist = 'l1'
dist = 'linf'
search_space = 'sub'


mooodel = 'gcn2012'
#   'gcn2012'  'gcn2007' 'liw2007'  'liw2012'

image_iter = 0


for i in range(11,23):
    print('This is num',i,'sample\n')

    inp = "./data/voc2012/VOCdevkit/VOC2012/JPEGImages/2012_0041{}.jpg".format(i)

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
        # print(pre)
        # print((pre == tar_label).all())

        return (pre == tar_label).all()

    ###############################################################

    def find_random_adversarial(image, epsilon=1000):

        num_calls = 1

        step = 0.02
        perturbed = x_0

        while is_adversarial(perturbed, tar_label) == 0:
            # print(is_adversarial(perturbed, tar_label))

            pert = torch.randn([1,3,448,448])
            pert = pert.to(device)

            perturbed = image + num_calls*step* pert
            perturbed = clip_image_values(perturbed, lb, ub)
            perturbed = perturbed.to(device)
            num_calls += 1
            if num_calls>100:
                break

        return perturbed, num_calls

    ###############################################################

    def bin_search(x_0, x_random, tol):

        num_calls = 0
        adv = x_random
        cln = x_0
        while True:

            mid = (cln + adv) / 2.0
            num_calls += 1

            if is_adversarial(mid, tar_label):
                adv = mid
            else:
                cln = mid

            if torch.norm(adv-cln).cpu().numpy()<tol and is_adversarial(adv, tar_label):
                break
            if num_calls >200 and is_adversarial(adv, tar_label):
                break
            if num_calls >205:
                print('jiejin bian jie xb')
                break
        return adv, num_calls

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

        epsilon = 10

        num_calls = 1
        perturbed = x_0

        if dist == 'l1' or dist == 'l2':

            grads = grad


        if dist == 'linf':

            grads = torch.sign(grad)/torch.norm(grad)

        while is_adversarial(perturbed, tar_label) == 0:

            perturbed = x_0 + (num_calls*epsilon* grads[0])
            perturbed = clip_image_values(perturbed, lb, ub)

            num_calls += 1

            if num_calls > 300:
                break
        if num_calls > 300:
            perturbed1,nnnuu=bin_search(x_0, x_b, tol)
            print('zheshi yi ge jie jin bian jie de  dui kang yang ben dian ')
            return perturbed1, num_calls+nnnuu, epsilon*num_calls

        return perturbed, num_calls, epsilon*num_calls

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
            if eps/10 >100:
                print("xadv jie jin bian jie")
                # x_adv = x_b

            print('chuan yue bian jie')
            q_num = q_num + qs
            x_adv, bin_query = bin_search(x_0, x_adv, tol)
            print('zhao dao bian jie dian')


            q_num = q_num + bin_query

            x_b = x_adv

            t2 = time.time()
            x_adv_inv = inv_tf(x_adv.cpu().numpy()[0,:,:,:].squeeze(), mean, std)

            if dist == 'l1' or dist == 'l2':
                dp = 'l2'
                norm_p = linalg.norm(x_adv_inv-image_fb)
            if dist == 'linf':
                dp = dist
                norm_p = np.max(abs(x_adv_inv-image_fb))

            if verbose_control == 'Yes':
                message = ' (took {:.5f} seconds)'.format(t2 - t1)
                print('iteration -> ' + str(i) + str(message) + '     -- ' + dp + ' norm is -> ' + str(norm_p))


        x_adv = clip_image_values(x_adv, lb, ub)

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
        # print(estimate_batch)
        ##
        # print(sub_basis_torch.shape)
        ## torch.Size([200704, 5625])
        EstNoise = SubNoise(estimate_batch, sub_basis_torch).cuda()
        random_vectors = EstNoise()
        random_vectors_np = random_vectors.cpu().numpy()



    ###############################################################
    #####
    # Models


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
    net = get_model(mooodel).eval()
    if torch.cuda.is_available():
        net = net.cuda()

    ###################################

    ##
    ## Load Image and Resize
    t11 = time.time()
    im_orig = Image.open(inp)
    im_sz = 448
    im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)

    image_fb = load_image(im_orig, data_format='channels_last')
    image_fb = image_fb / 255.  # because our model expects values in [0, 1]

    image_fb_first = load_image(im_orig, data_format='channels_first')
    image_fb_first = image_fb_first / 255.

    pred = (net(torch.tensor(image_fb_first).unsqueeze(0).cuda()) >= 0.5) + 0
    # print("clean label ",pred)

    # Bounds for Validity and Perceptibility
    delta = 255
    lb, ub = valid_bounds(im_orig, delta)


        # Transform data

    im = transforms.Compose([
        transforms.CenterCrop(448),
        transforms.ToTensor()])(im_orig)


    lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
    ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)

    im_deepfool = im.to(device)
    lb = lb[None, :, :, :].to(device)
    ub = ub[None, :, :, :].to(device)
    # print('lb .ub',lb,ub)
    # print(torch.max(lb),torch.min(lb))
    # print(torch.max(ub),torch.min(ub))

    x_0 = im[None, :, :, :].to(device)
    x_0_np = x_0.cpu().numpy()

    pred11 = (net(x_0) >= 0.5) + 0

    now_label = np.asarray(pred11.cpu())
    print('clean img labels:',now_label)
    tar_label = np.zeros((1,20))



    if (now_label == tar_label).all():
        print('Already missclassified ... Lets try another one!')

    else:
        image_iter = image_iter + 1

        x0_inverse = inv_tf(x_0.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
        dif_norm = linalg.norm(x0_inverse-image_fb)


    ###################################

        print('generate ini adv')
        x_random, query_random_1 = find_random_adversarial(x_0, epsilon=100)
        a = x_random

        # x_rnd_inverse = inv_tf(x_random.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
        # norm_rnd_inv = linalg.norm(x_rnd_inverse-image_fb)

        # print('x_random de zui dazhi', np.max(np.asarray(x_random.cpu())))


        print('is adversarial ?',is_adversarial(x_random, tar_label))

        # label_random = torch.argmax(net.forward(Variable(x_random, requires_grad=True)).data).item()

        print('Binary search \n')
        # Binary search

        x_boundary, query_binsearch_2 = bin_search(x_0, x_random, tol)

        if query_binsearch_2>205:
            break

        print('Binary search end\n')



        ###test is adv of xb
        # print(x_boundary.shape)
        # predict_xb = (net(x_boundary) >= 0.5) + 0
        # prexb = np.asarray(predict_xb.cpu())
        # print(prexb)
        # print((prexb == tar_label).all())

        x_b = x_boundary

        # Norm_rnd = torch.norm(x_0-x_boundary)
        # x_bin_inverse = inv_tf(x_boundary.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
        #
        # norm_bin_rnd = linalg.norm(x_bin_inverse-image_fb)
        #
        # x_rnd_BA = np.swapaxes(x_bin_inverse, 0, 2)
        # x_rnd_BA = np.swapaxes(x_rnd_BA, 1, 2)

        # is_adversarial(x_boundary, tar_label)
        #
        # label_boundary = torch.argmax(net.forward(Variable(x_boundary, requires_grad=True)).data).item()

        query_rnd = query_binsearch_2 + query_random_1

        # print('query_rnd',query_rnd)
        ###################################
        # Run over iterations
        iteration = round(Q_max/300)
        q_opt_it = int(Q_max  - (iteration)*25)
        q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, mu )
        q_opt_it = int(Q_max  - (iterate)*25)
        q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, mu )
        print('#################################################################')
        print('Start: The GeoDA will be run for:' + ' Iterations = ' + str(iterate) + ', Query = ' + str(Q_max) + ', Norm = ' + str(dist)+ ', Space = ' + str(search_space) )
        print('#################################################################')

        t3 = time.time()
        x_adv, query_o, gradient = GeoDA(x_b, iterate, q_opt_iter)

        # print(x_adv.shape)
        # print(type(x_adv))
        # p_x_adv = (net(x_adv) >= 0.5) + 0
        # p_x_adv111 = np.asarray(p_x_adv.cpu())
        # print('save pic{}'.format(i))
        #
        # imgadv = np.transpose(x_adv.cpu().numpy().squeeze(), (1, 2, 0))
        # imgadv = Image.fromarray(np.uint8(imgadv * 255))
        # imgadv.save("./adv_save/{}_adv.png".format(i))



        t4 = time.time()
        message = ' took {:.5f} seconds'.format(t4 - t3)
        qmessage = ' with query = ' + str(query_o + query_rnd)

        x_opt_inverse = inv_tf(x_adv.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
        # norm_inv_opt = linalg.norm(x_opt_inverse-image_fb)

        print('#################################################################')
        print('End: The GeoDA algorithm' + message + qmessage )
        print('#################################################################')

        if dist == 'l2' or dist == 'linf':
            adv_label = torch.argmax(net.forward(Variable(x_adv, requires_grad=True)).data).item()
            # str_label_adv = get_label(labels[int(adv_label)].split(',')[0])
            pert_norm = abs(x_opt_inverse-image_fb)/np.linalg.norm(abs(x_opt_inverse-image_fb))
            pert_norm_abs = (x_opt_inverse-image_fb)/np.linalg.norm((x_opt_inverse-image_fb))

            qizhi = 0
            for k in range(50):
                if qizhi == 0:
                    pertimage = image_fb + 10*k*pert_norm_abs
                    aaaadsdafwegg = pertimage.astype('float32')
                    aaaadsdafwegg = torch.tensor(np.transpose(aaaadsdafwegg, (2, 0, 1))).unsqueeze(0).cuda()
                    print('{} is adv'.format(k),is_adversarial(aaaadsdafwegg, tar_label))
                    if is_adversarial(aaaadsdafwegg, tar_label) :
                        print('is it adv?',is_adversarial(aaaadsdafwegg, tar_label))
                        pic = aaaadsdafwegg.cpu().numpy().squeeze()
                        pic = np.transpose(pic, (1,2,0))
                        print(pic.shape)
                        imgggg = Image.fromarray(np.uint8(pic * 255.))
                        imgggg.save("./gcn20123b/{}_adv.png".format(i))
                        qizhi = 1
                        break
                    # elif k >10:
                    #     pertimage = image_fb + 10*k * pert_norm_abs
                    #     aaaadsdafwegg = pertimage.astype('float32')
                    #     aaaadsdafwegg = torch.tensor(np.transpose(aaaadsdafwegg, (2, 0, 1))).unsqueeze(0).cuda()
                    #     print('{} is adv'.format(k), is_adversarial(aaaadsdafwegg, tar_label))
                    #     if is_adversarial(aaaadsdafwegg, tar_label):
                    #         pic = aaaadsdafwegg.cpu().numpy().squeeze()
                    #         pic = np.transpose(pic, (1, 2, 0))
                    #         print(pic.shape)
                    #         imgggg = Image.fromarray(np.uint8(pic * 255.))
                    #         imgggg.save("./adv_save/{}_adv.png".format(i))
                    #         break




            #
            # print(pertimage.shape)
            # print(type(pertimage))
            # imgggg = Image.fromarray(np.uint8(pertimage*255.))
            # imgggg.save("./adv_save/{}_adv.png".format(i))
for i in range(72,80):
    print('This is num',i,'sample\n')

    inp = "./data/voc2012/VOCdevkit/VOC2012/JPEGImages/2012_0041{}.jpg".format(i)

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
        # print(pre)
        # print((pre == tar_label).all())

        return (pre == tar_label).all()

    ###############################################################

    def find_random_adversarial(image, epsilon=1000):

        num_calls = 1

        step = 0.02
        perturbed = x_0

        while is_adversarial(perturbed, tar_label) == 0:
            # print(is_adversarial(perturbed, tar_label))

            pert = torch.randn([1,3,448,448])
            pert = pert.to(device)

            perturbed = image + num_calls*step* pert
            perturbed = clip_image_values(perturbed, lb, ub)
            perturbed = perturbed.to(device)
            num_calls += 1
            if num_calls>100:
                break

        return perturbed, num_calls

    ###############################################################

    def bin_search(x_0, x_random, tol):

        num_calls = 0
        adv = x_random
        cln = x_0
        while True:

            mid = (cln + adv) / 2.0
            num_calls += 1

            if is_adversarial(mid, tar_label):
                adv = mid
            else:
                cln = mid

            if torch.norm(adv-cln).cpu().numpy()<tol and is_adversarial(adv, tar_label):
                break
            if num_calls >200 and is_adversarial(adv, tar_label):
                break
            if num_calls >205:
                print('jiejin bian jie xb')
                break
        return adv, num_calls

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

        epsilon = 10

        num_calls = 1
        perturbed = x_0

        if dist == 'l1' or dist == 'l2':

            grads = grad


        if dist == 'linf':

            grads = torch.sign(grad)/torch.norm(grad)

        while is_adversarial(perturbed, tar_label) == 0:

            perturbed = x_0 + (num_calls*epsilon* grads[0])
            perturbed = clip_image_values(perturbed, lb, ub)

            num_calls += 1

            if num_calls > 300:
                break
        if num_calls > 300:
            perturbed1,nnnuu=bin_search(x_0, x_b, tol)
            print('zheshi yi ge jie jin bian jie de  dui kang yang ben dian ')
            return perturbed1, num_calls+nnnuu, epsilon*num_calls

        return perturbed, num_calls, epsilon*num_calls

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
            if eps/10 >100:
                print("xadv jie jin bian jie")
                # x_adv = x_b

            print('chuan yue bian jie')
            q_num = q_num + qs
            x_adv, bin_query = bin_search(x_0, x_adv, tol)
            print('zhao dao bian jie dian')


            q_num = q_num + bin_query

            x_b = x_adv

            t2 = time.time()
            x_adv_inv = inv_tf(x_adv.cpu().numpy()[0,:,:,:].squeeze(), mean, std)

            if dist == 'l1' or dist == 'l2':
                dp = 'l2'
                norm_p = linalg.norm(x_adv_inv-image_fb)
            if dist == 'linf':
                dp = dist
                norm_p = np.max(abs(x_adv_inv-image_fb))

            if verbose_control == 'Yes':
                message = ' (took {:.5f} seconds)'.format(t2 - t1)
                print('iteration -> ' + str(i) + str(message) + '     -- ' + dp + ' norm is -> ' + str(norm_p))


        x_adv = clip_image_values(x_adv, lb, ub)

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
        # print(estimate_batch)
        ##
        # print(sub_basis_torch.shape)
        ## torch.Size([200704, 5625])
        EstNoise = SubNoise(estimate_batch, sub_basis_torch).cuda()
        random_vectors = EstNoise()
        random_vectors_np = random_vectors.cpu().numpy()



    ###############################################################
    #####
    # Models


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
    net = get_model(mooodel).eval()
    if torch.cuda.is_available():
        net = net.cuda()

    ###################################

    ##
    ## Load Image and Resize
    t11 = time.time()
    im_orig = Image.open(inp)
    im_sz = 448
    im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)

    image_fb = load_image(im_orig, data_format='channels_last')
    image_fb = image_fb / 255.  # because our model expects values in [0, 1]

    image_fb_first = load_image(im_orig, data_format='channels_first')
    image_fb_first = image_fb_first / 255.

    pred = (net(torch.tensor(image_fb_first).unsqueeze(0).cuda()) >= 0.5) + 0
    # print("clean label ",pred)

    # Bounds for Validity and Perceptibility
    delta = 255
    lb, ub = valid_bounds(im_orig, delta)


        # Transform data

    im = transforms.Compose([
        transforms.CenterCrop(448),
        transforms.ToTensor()])(im_orig)


    lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
    ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)

    im_deepfool = im.to(device)
    lb = lb[None, :, :, :].to(device)
    ub = ub[None, :, :, :].to(device)
    # print('lb .ub',lb,ub)
    # print(torch.max(lb),torch.min(lb))
    # print(torch.max(ub),torch.min(ub))

    x_0 = im[None, :, :, :].to(device)
    x_0_np = x_0.cpu().numpy()

    pred11 = (net(x_0) >= 0.5) + 0

    now_label = np.asarray(pred11.cpu())
    print('clean img labels:',now_label)
    tar_label = np.zeros((1,20))



    if (now_label == tar_label).all():
        print('Already missclassified ... Lets try another one!')

    else:
        image_iter = image_iter + 1

        x0_inverse = inv_tf(x_0.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
        dif_norm = linalg.norm(x0_inverse-image_fb)


    ###################################

        print('generate ini adv')
        x_random, query_random_1 = find_random_adversarial(x_0, epsilon=100)
        a = x_random

        # x_rnd_inverse = inv_tf(x_random.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
        # norm_rnd_inv = linalg.norm(x_rnd_inverse-image_fb)

        # print('x_random de zui dazhi', np.max(np.asarray(x_random.cpu())))


        print('is adversarial ?',is_adversarial(x_random, tar_label))

        # label_random = torch.argmax(net.forward(Variable(x_random, requires_grad=True)).data).item()

        print('Binary search \n')
        # Binary search

        x_boundary, query_binsearch_2 = bin_search(x_0, x_random, tol)

        if query_binsearch_2>205:
            break

        print('Binary search end\n')



        ###test is adv of xb
        # print(x_boundary.shape)
        # predict_xb = (net(x_boundary) >= 0.5) + 0
        # prexb = np.asarray(predict_xb.cpu())
        # print(prexb)
        # print((prexb == tar_label).all())

        x_b = x_boundary

        # Norm_rnd = torch.norm(x_0-x_boundary)
        # x_bin_inverse = inv_tf(x_boundary.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
        #
        # norm_bin_rnd = linalg.norm(x_bin_inverse-image_fb)
        #
        # x_rnd_BA = np.swapaxes(x_bin_inverse, 0, 2)
        # x_rnd_BA = np.swapaxes(x_rnd_BA, 1, 2)

        # is_adversarial(x_boundary, tar_label)
        #
        # label_boundary = torch.argmax(net.forward(Variable(x_boundary, requires_grad=True)).data).item()

        query_rnd = query_binsearch_2 + query_random_1

        # print('query_rnd',query_rnd)
        ###################################
        # Run over iterations
        iteration = round(Q_max/300)
        q_opt_it = int(Q_max  - (iteration)*25)
        q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, mu )
        q_opt_it = int(Q_max  - (iterate)*25)
        q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, mu )
        print('#################################################################')
        print('Start: The GeoDA will be run for:' + ' Iterations = ' + str(iterate) + ', Query = ' + str(Q_max) + ', Norm = ' + str(dist)+ ', Space = ' + str(search_space) )
        print('#################################################################')

        t3 = time.time()
        x_adv, query_o, gradient = GeoDA(x_b, iterate, q_opt_iter)

        # print(x_adv.shape)
        # print(type(x_adv))
        # p_x_adv = (net(x_adv) >= 0.5) + 0
        # p_x_adv111 = np.asarray(p_x_adv.cpu())
        # print('save pic{}'.format(i))
        #
        # imgadv = np.transpose(x_adv.cpu().numpy().squeeze(), (1, 2, 0))
        # imgadv = Image.fromarray(np.uint8(imgadv * 255))
        # imgadv.save("./adv_save/{}_adv.png".format(i))



        t4 = time.time()
        message = ' took {:.5f} seconds'.format(t4 - t3)
        qmessage = ' with query = ' + str(query_o + query_rnd)

        x_opt_inverse = inv_tf(x_adv.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
        # norm_inv_opt = linalg.norm(x_opt_inverse-image_fb)

        print('#################################################################')
        print('End: The GeoDA algorithm' + message + qmessage )
        print('#################################################################')

        if dist == 'l2' or dist == 'linf':
            adv_label = torch.argmax(net.forward(Variable(x_adv, requires_grad=True)).data).item()
            # str_label_adv = get_label(labels[int(adv_label)].split(',')[0])
            pert_norm = abs(x_opt_inverse-image_fb)/np.linalg.norm(abs(x_opt_inverse-image_fb))
            pert_norm_abs = (x_opt_inverse-image_fb)/np.linalg.norm((x_opt_inverse-image_fb))

            qizhi = 0
            for k in range(50):
                if qizhi == 0:
                    pertimage = image_fb + 10*k*pert_norm_abs
                    aaaadsdafwegg = pertimage.astype('float32')
                    aaaadsdafwegg = torch.tensor(np.transpose(aaaadsdafwegg, (2, 0, 1))).unsqueeze(0).cuda()
                    # print('{} is adv'.format(k),is_adversarial(aaaadsdafwegg, tar_label))
                    if is_adversarial(aaaadsdafwegg, tar_label) :
                        print('is it adv?',is_adversarial(aaaadsdafwegg, tar_label))
                        pic = aaaadsdafwegg.cpu().numpy().squeeze()
                        pic = np.transpose(pic, (1,2,0))
                        print(pic.shape)
                        imgggg = Image.fromarray(np.uint8(pic * 255.))
                        imgggg.save("./gcn20123b/{}_adv.png".format(i))
                        qizhi = 1
                        break
                    # elif k >10:
                    #     pertimage = image_fb + 10*k * pert_norm_abs
                    #     aaaadsdafwegg = pertimage.astype('float32')
                    #     aaaadsdafwegg = torch.tensor(np.transpose(aaaadsdafwegg, (2, 0, 1))).unsqueeze(0).cuda()
                    #     print('{} is adv'.format(k), is_adversarial(aaaadsdafwegg, tar_label))
                    #     if is_adversarial(aaaadsdafwegg, tar_label):
                    #         pic = aaaadsdafwegg.cpu().numpy().squeeze()
                    #         pic = np.transpose(pic, (1, 2, 0))
                    #         print(pic.shape)
                    #         imgggg = Image.fromarray(np.uint8(pic * 255.))
                    #         imgggg.save("./adv_save/{}_adv.png".format(i))
                    #         break




            #
            # print(pertimage.shape)
            # print(type(pertimage))
            # imgggg = Image.fromarray(np.uint8(pertimage*255.))
            # imgggg.save("./adv_save/{}_adv.png".format(i))



