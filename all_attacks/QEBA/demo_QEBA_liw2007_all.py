from __future__ import absolute_import, division, print_function

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch

print(torch.cuda.is_available())
print(torch.__version__)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.fft
import argparse

from tqdm import tqdm
import time
import requests
import torchvision
import cv2
from ml_gcn_model.models import gcn_resnet101_attack
from ml_gcn_model.voc import Voc2007Classification
from ml_gcn_model.util import Warp
import torchvision.transforms as transforms
from ml_gcn_model.voc import write_object_labels_csv

from ml_liw_model.models import inceptionv3_attack
from ml_liw_model.voc import Voc2012Classification
from ml_gcn_model.util import Warp
from ml_liw_model.voc import write_object_labels_csv

import torch.nn as nn
import numpy as np
from numpy import linalg
import imageio
from PIL import Image
from QEBA import qeba
import numpy as np
import argparse




def get_model(x):
    num_classes = 20
    # load torch model
    if x == 'gcn2007':
        print('load gcn2007 model success')
        model = gcn_resnet101_attack(num_classes=num_classes,
                                     t=0.4,
                                     adj_file='./data/voc2007/voc_adj.pkl',
                                     word_vec_file='./data/voc2007/voc_glove_word2vec.pkl',
                                     save_model_path='./checkpoint/mlgcn/voc2007/voc_checkpoint.pth.tar')
    elif x == 'gcn2012':
        print('load gcn2012 model success')
        model = gcn_resnet101_attack(num_classes=num_classes,
                                     t=0.4,
                                     adj_file='./data/voc2012/voc_adj.pkl',
                                     word_vec_file='./data/voc2012/voc_glove_word2vec.pkl',
                                     save_model_path='./checkpoint/mlgcn/voc2012/voc_checkpoint.pth.tar')
    elif x == 'liw2007':
        print('load liw2007 model success')
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2007/model_best.pth.tar')
    elif x == 'liw2012':
        print('load liw2012 model success')
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2012/model_best.pth.tar')

    return model.eval()


###########################################
def evaluate_adv(state):
    model = state['model']
    y_target = state['y_target']
    adv_folder_path = state['output_folder']
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x: int(x[0:-6]))
    ori_folder_path = state['ori_folder']
    ori_file_list = os.listdir(ori_folder_path)
    ori_file_list.sort(key=lambda x: int(x[0:-4]))
    adv = []
    for f in adv_file_list:
        a = np.asarray(Image.open(adv_folder_path + f)) / 255.
        adv.extend(np.expand_dims(a, axis=0))
    adv = np.asarray(adv)

    new_adv = []

    for img in adv:
        img = Image.fromarray(np.uint8(img * 255))
        crop = transforms.Resize(448)
        img = crop(img)
        img = np.asarray(img)
        img = img.astype('float32')
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        new_adv.append(img)
    new_adv = np.asarray(new_adv)

    adv = new_adv

    ori = []
    for f in ori_file_list:
        a = np.asarray(Image.open(ori_folder_path + f).resize((448, 448))) / 255.
        ori.extend(np.expand_dims(a, axis=0))
    ori = np.asarray(ori)
    new_ori = []
    for img in ori:
        img = Image.fromarray(np.uint8(img * 255))
        crop = transforms.Resize(448)
        img = crop(img)
        img = np.asarray(img)
        img = img.astype('float32')
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        new_ori.append(img)
    new_ori = np.asarray(new_ori)

    ori = new_ori

    dl1 = torch.utils.data.DataLoader(adv,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=4)

    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
    ])
    # ori.transform = data_transforms

    dl2 = torch.utils.data.DataLoader(ori,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=4)
    dl2 = tqdm(dl2, desc='ADV')

    adv_output = []
    norm_1 = []
    norm = []
    max_r = []
    mean_r = []
    rmsd = []

    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            if torch.cuda.is_available():
                batch_adv_x = batch_adv_x.cuda()
            adv_output.extend(model(batch_adv_x).cpu().numpy())
            batch_adv_x = batch_adv_x.cpu().numpy()
            batch_test_x = batch_test_x[0][0].cpu().numpy()
            batch_r = (batch_adv_x - batch_test_x)
            batch_r_255 = ((batch_adv_x / 2 + 0.5) * 255) - ((batch_test_x / 2 + 0.5) * 255)
            batch_norm = [np.linalg.norm(r.flatten()) for r in batch_r]
            batch_rmsd = [np.sqrt(np.mean(np.square(r))) for r in batch_r_255]
            norm.extend(batch_norm)
            rmsd.extend(batch_rmsd)

            norm_1.extend(np.sum(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            max_r.extend(np.max(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            mean_r.extend(np.mean(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
    adv_output = np.asarray(adv_output)
    adv_pred = adv_output.copy()
    adv_pred[adv_pred >= (0.5 + 0)] = 1
    adv_pred[adv_pred < (0.5 + 0)] = -1

    # 原始 # 按照2范数来选择
    adv_pred_match_target = (np.logical_and(np.all((adv_pred == y_target), axis=1), np.asarray(norm) < (77.6 * 3))) + 0
    # adv_pred_match_target = (np.asarray(norm) < (77.6 * 3)) + 0

    attack_fail_idx = np.argwhere(adv_pred_match_target == 0).flatten()

    np.save('attack_fail_idx.npy', attack_fail_idx)
    norm = np.asarray(norm)

    max_r = np.asarray(max_r).round(4)
    mean_r = np.asarray(mean_r).round(4)
    rmsd = np.asarray(rmsd).round(4)
    norm = np.delete(norm, attack_fail_idx, axis=0).round(4)
    max_r = np.delete(max_r, attack_fail_idx, axis=0).round(4)
    norm_1 = np.delete(norm_1, attack_fail_idx, axis=0).round(4)
    mean_r = np.delete(mean_r, attack_fail_idx, axis=0).round(4)
    rmsd = np.delete(rmsd, attack_fail_idx, axis=0).round(4)

    # from utils import evaluate_metrics
    metrics = dict()
    y_target[y_target == -1] = 0
    # metrics['ranking_loss'] = evaluate_metrics.label_ranking_loss(y_target, adv_output)
    # metrics['average_precision'] = evaluate_metrics.label_ranking_average_precision_score(y_target, adv_output)
    # metrics['auc'] = evaluate_metrics.roc_auc_score(y_target, adv_output)
    # auc曲线
    metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
    metrics['norm'] = np.mean(norm).round(4)
    metrics['norm_1'] = np.mean(norm_1).round(4)
    metrics['rmsd'] = np.mean(rmsd).round(4)
    metrics['max_r'] = np.mean(max_r).round(4)
    metrics['mean_r'] = np.mean(mean_r).round(4)
    print()
    print(metrics)
    print('can we here?')
    return metrics


#########################################
def testresult(model, ori_folder, output_folder, adv_num):
    print("Load Data from ori_folder")
    X = []
    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
    ])
    ori_imge = os.listdir(ori_folder)
    ori_imge.sort(key=lambda x: int(x[-7:-4]))
    with torch.no_grad():
        for img in ori_imge:
            a = Image.open(os.path.join(ori_folder, img))
            a = data_transforms(a).unsqueeze(0)
            X.append(a)
    X = torch.cat(X, 0)
    # X = torch.cat(X, 0) / 255
    # y = model(X).argmax(1)
    ####################################################
    print("Evaluate data in output_folder")
    if torch.cuda.is_available():
        model = model.cuda()
        X = X.cuda()
        # y = y.cuda()
    y_target = np.zeros((adv_num, 20)) - 1
    # y_target = np.load('./y_t.npy')
    # print(y_target.shape)
    state = {'model': model,
             'ori_folder': ori_folder,
             'output_folder': output_folder,
             'y_target': y_target
             }

    return evaluate_adv(state)
    # pre_adv(state)



def gen_adv_file(model, target_type, ori_img_path,npy_path):
    print("generiting target file…")

    advlist = os.listdir(ori_img_path)
    advlist.sort(key=lambda x: int(x[-7:-4]))


    adv = []
    for f in advlist:
        a = np.asarray(Image.open(ori_img_path + f)) / 255.
        adv.extend(np.expand_dims(a, axis=0))
    adv = np.asarray(adv)

    new_adv = []

    for img in adv:
        img = Image.fromarray(np.uint8(img * 255)).resize((448, 448))
        crop = transforms.Resize(448)
        img = crop(img)
        img = np.asarray(img)
        img = img.astype('float32')
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        new_adv.append(img)
    new_adv = np.asarray(new_adv)

    adv = new_adv

    dl1 = torch.utils.data.DataLoader(adv,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=4)

    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
    ])

    output = []
    y=[]

    with torch.no_grad():
        for adv_x in dl1:
            y_ori  =model(adv_x.cuda()).cpu().numpy()

            y_ori= (y_ori>0.5)+0
            # print(y_ori)
            y.extend(y_ori)
            y_target = get_target_label(y_ori, target_type)
            # 'random_case''extreme_case''person_reduction''sheep_augmentation''hide_single''hide_all'
            # print(y_target)
            output.extend(y_target)

    # print(output)
    # print(y)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    np.save(os.path.join(npy_path, 'y_target.npy'), output)
    np.save(os.path.join(npy_path, 'y.npy'), y)
    print('npy has saved')

def get_target_label(yori, target_type):
    '''
    :param y: numpy, y in {0, 1}
    :param A: list, label index that we want to reverse
    :param C: list, label index that we don't care
    :return:
    '''
    y = yori.copy()
    # o to -1

    y[y == 0] = -1
    # print(y)

    if (y == -1).all():
        y = yori
    else:
        if target_type == 'random_case':
            for i, y_i in enumerate(y):
                pos_idx = np.argwhere(y_i == 1).flatten()
                neg_idx = np.argwhere(y_i == -1).flatten()
                pos_idx_c = np.random.choice(pos_idx)
                neg_idx_c = np.random.choice(neg_idx)
                y[i, pos_idx_c] = -y[i, pos_idx_c]
                y[i, neg_idx_c] = -y[i, neg_idx_c]
        elif target_type == 'extreme_case':
            y = -y
        elif target_type == 'person_reduction':
            # person in 14 col
            y[:, 14] = -1
        elif target_type == 'sheep_augmentation':
            # sheep in 16 col
            y[:, 16] = 1
        elif target_type == 'hide_single':
            for i, y_i in enumerate(y):
                pos_idx = np.argwhere(y_i == 1).flatten()
                pos_idx_c = np.random.choice(pos_idx)
                y[i, pos_idx_c] = -y[i, pos_idx_c]
        elif target_type == 'hide_all':
                y[y == 1] = -1

    y[y == -1] = 0

    return y




def attack(args, numiter,model1, img_path):
    model = model1
    targets = 0

    sample448 = np.array(Image.open(img_path).resize((448, 448)).convert('RGB')) / 255.
    # sample = img
    pic = np.array(Image.open(img_path).resize((224, 224)).convert('RGB'))
    data = cv2.resize(pic, dsize=None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    sample224 =data/255.

    perturbed,query_num = qeba(model,
                     sample448,
                     sample224,
                     clip_max=1,
                     clip_min=0,
                     constraint=args.constraint,
                     num_iterations=numiter,
                     gamma=1.0,
                     target_label=None,
                     target_image=None,
                     stepsize_search=args.stepsize_search,
                     max_num_evals=10,
                     init_num_evals=10)

    return perturbed,query_num


if __name__ == '__main__':

    torch.manual_seed(998)
    torch.cuda.manual_seed(998)
    np.random.seed(998)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--constraint', type=str,
                        choices=['l2', 'linf'],
                        default='l2')
    parser.add_argument('--attack_type', type=str,
                        choices=['targeted', 'untargeted'],
                        default='untargeted')
    parser.add_argument('--num_samples', type=int,
                        default=10)
    parser.add_argument('--num_iterations', type=int,
                        default=10)
    parser.add_argument('--stepsize_search', type=str,
                        choices=['geometric_progression', 'grid_search'],
                        default='geometric_progression')

    args = parser.parse_args()
    dict_a = vars(args)

    # model = ML_GCN2007()
    model = get_model('liw2007').to(device)

    npy_path = './QEBA_data/npy/liw2007/hideall/'

    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    adv_save_path = './QEBA_data/adv_save/voc2007/mlliw/hideall/'
    if not os.path.exists(adv_save_path):
        os.makedirs(adv_save_path)


    ori_img_path = './ori_test_2007/'


    ori_448_path = './QEBA_data/ori_save/voc2007/'
    if not os.path.exists(ori_448_path):
        os.makedirs(ori_448_path)

    ############# ATTACK TYPE ###############################
    gen_adv_file(model, 'hide_all', ori_img_path, npy_path)
    ###   'hide_single'  'hide_all'     'random_case'   'extreme_case'   'person_reduction'  'sheep_augmentation'



    adv_file_list = os.listdir(ori_img_path)
    adv_file_list.sort(key=lambda x: int(x[-7:-4]))


    with torch.no_grad():
        for num in [ 30000]:
            print('NOW  ITER = 300')
            print(num)

            ii = 1
            for f in adv_file_list:
                if ii > 20:
                    break

                test = Image.open(os.path.join(ori_img_path,f)).resize((448, 448)).convert('RGB')
                test.save(os.path.join(ori_448_path, '{m}.png'.format(m=ii)))
                print('This is num {} pic\n'.format(ii))

                x_adv,qnum =attack(args,int(num/100), model, os.path.join(ori_img_path,f))

                if not os.path.exists(os.path.join(adv_save_path, '{n}/'.format(n=num))):
                    os.makedirs(os.path.join(adv_save_path, '{n}/'.format(n=num)))

                imageio.imwrite(os.path.join(adv_save_path, '{n}/{m}_adv.png'.format(n=num, m=ii)), np.uint8(x_adv * 255.))

                ii+=1





    # adv = []
    # for f in adv_file_list:
    #     print(f)
    #     # 读取图像
    #     a = np.asarray(Image.open(adv_folder_path + f).resize((448, 448)) )/ 255.
    #
    #     # Image.open().resize((448, 448)).convert(
    #     #     'RGB'))
    #     print(a.shape)
    #     # 读取数组
    #     adv.extend(np.expand_dims(a, axis=0))
    # print(len(adv))
    # adv = np.asarray(adv)
    # new_adv = []
    # index_name = 1
    # for img in adv:
    #     # print(img.shape)
    #     # img = np.transpose(img, (1, 2, 0))
    #     img = Image.fromarray(np.uint8(img * 255))
    #     crop = transforms.Resize(448)
    #     img = crop(img)
    #     imageio.imwrite('./ori_save/voc2012/{n}.png'.format(n=index_name), img)
    #     index_name+=1
    #     img = np.asarray(img)
    #     img = img.astype('float32')
    #     img = img / 255
    #     # img = np.transpose(img, (2, 0, 1))
    #     new_adv.append(img)
    # new_adv = np.asarray(new_adv)
    #
    # adv = new_adv
    # ii = 1
    # for img in adv:
    #     print('This is num {} pic\n'.format(ii))
    #     print(img.shape)
    #     attack(args, model, img, ii)
    #     ii+=1

