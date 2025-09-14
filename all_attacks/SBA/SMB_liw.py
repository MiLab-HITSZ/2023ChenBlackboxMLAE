import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch

print('Use GPU {}'.format(torch.cuda.is_available()),torch.__version__)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.fft
import argparse

from tqdm import tqdm
import time
import requests
import torchvision

from ml_gcn_model.models import gcn_resnet101_attack
from ml_gcn_model.voc import Voc2007Classification
from ml_gcn_model.util import Warp
import torchvision.transforms as transforms
from ml_gcn_model.voc import write_object_labels_csv

from ml_liw_model.models import inceptionv3_attack
from ml_liw_model.voc import Voc2012Classification
from ml_gcn_model.util import Warp
from ml_liw_model.voc import write_object_labels_csv


import utils
import math
import random
import argparse
import os
import numpy as np
from PIL import Image

from simba import SimBA

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
# parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
parser.add_argument('--sampled_image_dir', type=str, default='save', help='directory to cache sampled images')
parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for parallel runs')
parser.add_argument('--num_iters', type=int, default=3000, help='maximum number of iterations, 0 for unlimited')
parser.add_argument('--log_every', type=int, default=20, help='log every n iterations')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
parser.add_argument('--freq_dims', type=int, default=14, help='dimensionality of 2D frequency space')
parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
parser.add_argument('--stride', type=int, default=7, help='stride for block order')
parser.add_argument('--targeted', action='store_true',default=True, help='perform targeted attack')
parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
parser.add_argument('--pic', type=int, default=1, help='pic name')
args = parser.parse_args()

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if not os.path.exists(args.sampled_image_dir):
    os.mkdir(args.sampled_image_dir)


def get_model(x):
    num_classes = 20
    # load torch model
    if x == 'gcn_2007':
        print('load gcn2007 model success')
        model = gcn_resnet101_attack(num_classes=num_classes,
                                     t=0.4,
                                     adj_file='./data/voc2007/voc_adj.pkl',
                                     word_vec_file='./data/voc2007/voc_glove_word2vec.pkl',
                                     save_model_path='./checkpoint/mlgcn/voc2007/voc_checkpoint.pth.tar')
    elif x == 'gcn_2012':
        print('load gcn2012 model success')
        model = gcn_resnet101_attack(num_classes=num_classes,
                                     t=0.4,
                                     adj_file='./data/voc2012/voc_adj.pkl',
                                     word_vec_file='./data/voc2012/voc_glove_word2vec.pkl',
                                     save_model_path='./checkpoint/mlgcn/voc2012/voc_checkpoint.pth.tar')
    elif x == 'liw_2007':
        print('load liw2007 model success')
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2007/model_best.pth.tar')
    elif x == 'liw_2012':
        print('load liw2012 model success')
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2012/model_best.pth.tar')

    return model.eval()


def evaluate_adv(state):
    model = state['model']
    samplenum = state['num']
    y_target = state['y_target']
    adv_folder_path = state['output_folder']
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x: int(x[0:-8]))
    ori_folder_path = state['ori_folder']
    ori_file_list = os.listdir(ori_folder_path)
    ori_file_list.sort(key=lambda x: int(x[0:-4]))
    adv = []

    advnum = 0
    for f in adv_file_list:
        if advnum>=samplenum:
            break
        a = np.asarray(Image.open(adv_folder_path + f)) / 255.
        adv.extend(np.expand_dims(a, axis=0))
        advnum+=1

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
    orinum =0
    for f in ori_file_list:
        if orinum>=samplenum:
            break
        a = np.asarray(Image.open(ori_folder_path + f).resize((448, 448))) / 255.
        ori.extend(np.expand_dims(a, axis=0))
        orinum +=1

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

    print(adv.shape,ori.shape)

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
    # dl2 = tqdm(dl2, desc='ADV')

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
    adv_pred[adv_pred < (0.5 + 0)] = 0
    # print(adv_pred.shape)
    # print(y_target)

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
    metrics['max_r'] = np.mean(max_r).round(4)
    metrics['mean_r'] = np.mean(mean_r).round(4)
    metrics['rmsd'] = np.mean(rmsd).round(4)
    print()
    print(metrics)
    print('can we here?')
    return metrics
#########################################
def testresult(model, ori_folder, output_folder, y_target,adv_num):
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
    # y_target = np.zeros((adv_num, 20)) - 1
    # y_target = np.load('./y_t.npy')
    # print(y_target.shape)
    y_target = y_target[0:adv_num,:]
    print(y_target.shape)
    state = {'model': model,
             'ori_folder': ori_folder,
             'output_folder': output_folder,
             'y_target': y_target,
             'num':adv_num
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

if __name__ == '__main__':

    torch.manual_seed(998)
    torch.cuda.manual_seed(998)
    np.random.seed(998)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # ###############    parm   ###################
    dataset = 'voc2007'
    # voc2007    voc2012  nuswide
    model_name='liw_2007'
    # 'gcn_2007'   'gcn_2012'  'liw_2007'  'liw_2012'
    # attack_type='random_case'
    #'hide_single'  'hide_all'     'random_case'   'extreme_case'   'person_reduction'  'sheep_augmentation'

    # # ###############   without change   ###################

    model = get_model(model_name).to(device)

    ori_img_path = './ori_{}/'.format(dataset)

    ori_448_path = './ori_{data}_448/'.format(data=dataset)
    if not os.path.exists(ori_448_path):
        os.makedirs(ori_448_path)



    for a_type in ['random_case','sheep_augmentation']:
        attack_type = a_type
        print('now is {}'.format(attack_type))

        #############      ADV save path     ###############################
        adv_save_path = './adv_save/{data}/{model_name}/{attack_type}/'.format(data=dataset, model_name=model_name,
                                                                               attack_type=attack_type)
        if not os.path.exists(adv_save_path):
            os.makedirs(adv_save_path)

        #############      NPY save path     ###############################
        npy_path = './npy/{model_name}/{attack_type}/'.format(model_name=model_name, attack_type=attack_type)
        if not os.path.exists(npy_path):
            os.makedirs(npy_path)

        #############      gener NPY    ###############################
        gen_adv_file(model, attack_type, ori_img_path, npy_path)
        ###   'hide_single'  'hide_all'     'random_case'   'extreme_case'   'person_reduction'  'sheep_augmentation'

        #############     load  NPY      ###############################
        y_t = np.load(os.path.join(npy_path, 'y_target.npy'))

        #############     test ADV    ###############################
        # testresult(model, ori_448_path, './adv_save/voc2007/mlgcn/hidesingle/3w/', y_t, 100)


        ################   ATTACK   ##################

        image_size = 448
        attacker = SimBA(model, 'voc', image_size)

        ori_image = os.listdir(ori_img_path)
        print(ori_img_path)
        ori_image.sort(key=lambda x: int(x[-7:-4]))

        for num_iters in [30000]:
            print('{} attack, queries = {}'.format(attack_type,num_iters))

            ii=1
            for f in ori_image:

                if ii >50:
                    break

                y_target = y_t[ii-1]
                # print(y_target)
                print(os.path.join(ori_img_path,f))
                test = Image.open(os.path.join(ori_img_path,f)).resize((448, 448)).convert('RGB')
                test.save(os.path.join(ori_448_path,'{m}.png'.format(m=ii)))

                print('This is num {} pic\n'.format(ii))
                # print(os.path.join(adv_folder_path,f))
                images = torch.tensor((np.array(test).transpose(2, 0, 1).astype('float32') / 255.).reshape(1, 3, 448, 448)
                                      ).to(device)

                adv = attacker.simba_single(images, y_target, num_iters=num_iters, epsilon=1, targeted=False)
                imgadv = Image.fromarray(np.uint8(adv.squeeze().cpu() * 255).transpose(1, 2, 0), 'RGB')

                if not os.path.exists(os.path.join(adv_save_path, '{n}/'.format(n=num_iters))):
                    os.makedirs(os.path.join(adv_save_path, '{n}/'.format(n=num_iters)))

                imgadv.save(os.path.join(adv_save_path,'{itr}/{m}_adv.png'.format(itr=num_iters,m=ii)))
                ii+=1

