import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
                                   save_model_path='./checkpoint/mlliw/coco/model_best.pth.tar')

    elif x == 'liwnus':
        model = inceptionv3_attack(num_classes=81,
                                   save_model_path='./checkpoint/mlliw/NUSWIDE/model_best.pth.tar')
    elif x == 'liw2007':
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2007/model_best.pth.tar')

    elif x == 'liw2012':
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2012/model_best.pth.tar')

    return model.eval()

###########################################
def evaluate_adv(state):
    model = state['model']
    y_target = state['y_target']
    adv_folder_path = state['output_folder']
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x: int(x[0:-8]))
    ori_folder_path = state['ori_folder']
    ori_file_list = os.listdir(ori_folder_path)


    # ori_file_list.sort(key=lambda x: int(x[-7:-4]))
    #coco
    ori_file_list.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # ori_file_list.sort(key=lambda x: int(x[0:6]))



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

    # 原始 # 按照2范数来选择
    adv_pred_match_target = (   np.logical_and(  np.logical_and(np.all((adv_pred == y_target), axis=1),np.asarray(norm) < (77.6 * 1))  ,np.asarray(norm) >(0.00)) ) + 0
    # adv_pred_match_target = (    np.logical_and(np.all((adv_pred == y_target), axis=1),np.asarray(norm) < (77.6 * 3))  ) + 0
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
def testresult(model, ori_folder, output_folder,y_target, adv_num):
    print("Load Data from ori_folder")
    X = []
    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
    ])
    ori_imge = os.listdir(ori_folder)


    # ori_imge.sort(key=lambda x: int(x[-7:-4]))
    # #coco
    ori_imge.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # ori_imge.sort(key=lambda x: int(x[0:6]))


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
    y_target = y_target[0:adv_num]
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

    # advlist.sort(key=lambda x: int(x[-7:-4]))
    # #coco
    advlist.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # advlist.sort(key=lambda x: int(x[0:6]))



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

    # ###############model = ML_GCN2007()###################
    m_ty = "mlgcn"
    modelname = 'gcncoco'
    attackt = "hide_single"
    datasetn = "coco"
    qnummmm = 3000

    model = get_model(modelname).to(device)


    npy_path = './npy/{}/{}/'.format(modelname,attackt)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    adv_save_path = './adv_save/{}/{}/{}/'.format(datasetn,m_ty,attackt)
    if not os.path.exists(adv_save_path):
        os.makedirs(adv_save_path)

    ori_img_path = './{}ori/'.format(datasetn)

    ori_448_path = './{}ori448/'.format(datasetn)
    if not os.path.exists(ori_448_path):
        os.makedirs(ori_448_path)

    adv_file_list = os.listdir(ori_img_path)

    # #voc
    # adv_file_list.sort(key=lambda x: int(x[-7:-4]))
    # #coco
    adv_file_list.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # adv_file_list.sort(key=lambda x: int(x[0:6]))


    gen_adv_file(model, attackt, ori_img_path, npy_path)

    y_t = np.load(os.path.join(npy_path,'y_target.npy'))

################### evaluate ##################
    # print(testresult(model, './ori_save/voc2007/', './adv_save/voc2007/mlliw/hideall/3w/', 100))
    ################ get data ###############
    image_size = 448

    # y_target = np.zeros([1, 20], dtype=float)



    attacker = SimBA(model, 'voc', image_size)




    for num_iters in [20000,2000]:
        print('{} attack, queries = {}'.format(attackt, num_iters))

        ii=1


        for f in adv_file_list:
            if ii >25:
                break
            elif ii>0:


                y_target = y_t[ii-1]
                # print(y_target)
                test = Image.open(os.path.join(ori_img_path,f)).resize((448, 448)).convert('RGB')

                test.save(os.path.join(ori_448_path, '{m}.png'.format(m=ii)))


                print('This is num {} pic\n'.format(ii))
                # print(os.path.join(adv_folder_path,f))
                images = torch.tensor((np.array(test).transpose(2, 0, 1).astype('float32') / 255.).reshape(1, 3, 448, 448)
                                      ).to(device)

                adv = attacker.simba_single(images, y_target, num_iters=num_iters, epsilon=0.3, targeted=False)

                imgadv = Image.fromarray(np.uint8(adv.squeeze().cpu() * 255.).transpose(1, 2, 0), 'RGB')

                if not os.path.exists(os.path.join(adv_save_path, '{n}/'.format(n=num_iters))):
                    os.makedirs(os.path.join(adv_save_path, '{n}/'.format(n=num_iters)))
                print('save?',os.path.join(adv_save_path,'{}/{m}_adv.png'.format(num_iters,m=ii)))
                imgadv.save( os.path.join(adv_save_path,'{}/{m}_adv.png'.format(num_iters,m=ii)))


            ii+=1
