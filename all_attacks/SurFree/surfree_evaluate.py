import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
print(torch.cuda.is_available())
print(torch.__version__)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import torch.fft
import torchvision.transforms as transforms
import numpy as np
import argparse
from tqdm import tqdm
import time
import requests
import torchvision

from PIL import Image
from torchvision import transforms as T
from torchvision.io import read_image
from surfree import SurFree

from ml_gcn_model.models import gcn_resnet101_attack
from ml_gcn_model.voc import Voc2007Classification
from ml_gcn_model.util import Warp
from ml_gcn_model.voc import write_object_labels_csv
from ml_liw_model.models import inceptionv3_attack
from ml_liw_model.voc import Voc2012Classification
from ml_gcn_model.util import Warp
from ml_liw_model.voc import write_object_labels_csv

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

    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", "-model", default="liwnus", help="modelname")
    parser.add_argument("--attacktype", "-attack", default="hideall", help="hideall, hidesingle")
    parser.add_argument("--config_path0", default="config_300.json",help="Configuration")
    parser.add_argument("--config_path1", default="config_3000.json", help="Configuration")
    parser.add_argument("--config_path2", default="config_30000.json", help="Configuration")
    parser.add_argument("--qnum", default="300",help="qnum")
## single##
    # parser.add_argument("--ini_path", "-ini", default="./hidesingle/GCN_2007_adv/", help="ini_path")
    parser.add_argument("--ori_folder", "-ori", default="./nuswide_ori/", help="ori folder")
    parser.add_argument("--ori_folder_save", "-ori_save", default="./nuswide_ori448/", help="ori folder")

###
    # parser.add_argument("--ori_folder", "-ori", default="./adv_save/gcn2007/ori/", help="ori folder")

    parser.add_argument("--output_folder", "-out", default="./adv_save/liwnus/", help="Output folder")


    parser.add_argument("--n_images", "-n", type=int, default=2, help="N images attacks")

    return parser.parse_args()



def gen_adv_file(model, target_type, ori_img_path, npy_path):
    print("generiting target file…")

    advlist = os.listdir(ori_img_path)

    # advlist.sort(key=lambda x: int(x[0:-8]))
    advlist.sort(key=lambda x: int(x[-7:-4]))
    # coco
    # advlist.sort(key=lambda x: int(x[-10:-4]))
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
    y = []

    with torch.no_grad():
        for adv_x in dl1:
            y_ori = model(adv_x.cuda()).cpu().numpy()

            y_ori = (y_ori > 0.5) + 0
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

###########################################
def evaluate_adv(state):
    model = state['model']
    samplenum = state['num']
    y_target = state['y_target']
    adv_folder_path = state['output_folder']
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x: int(x[0:-8]))
    ori_folder_path = state['ori_folder']
    ori_file_list = os.listdir(ori_folder_path)

    ori_file_list.sort(key=lambda x: int(x[-7:-4]))
    # coco
    # ori_file_list.sort(key=lambda x: int(x[-10:-4]))
    # ori_file_list.sort(key=lambda x: int(x[0:-8]))
    # nuswide
    # ori_file_list.sort(key=lambda x: int(x[0:6]))

    print(adv_file_list)
    print(ori_file_list)

    adv = []

    advnum = 0
    for f in adv_file_list:
        if advnum >= samplenum:
            break
        a = np.asarray(Image.open(adv_folder_path + f)) / 255.
        adv.extend(np.expand_dims(a, axis=0))
        advnum += 1

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
    orinum = 0
    for f in ori_file_list:
        if orinum >= samplenum:
            break
        a = np.asarray(Image.open(ori_folder_path + f).resize((448, 448))) / 255.
        ori.extend(np.expand_dims(a, axis=0))
        orinum += 1

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

    print(adv.shape, ori.shape)

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
    # adv_pred_match_target = (np.logical_and(
    #     np.logical_and(np.all((adv_pred == y_target), axis=1),
    #                    np.asarray(norm) < (77.6 * 1)), np.asarray(norm) > (0.00))) + 0
    print(norm)
    adv_pred_match_target = (np.asarray(norm) < (77.6*3)) + 0
    print(adv_pred_match_target)

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

    t =1.1647
    for i in ['norm','norm_1','max_r','mean_r','rmsd']:
        metrics['norm'] = np.mean(norm).round(4)  * t
        metrics['norm_1'] = np.mean(norm_1).round(4)* t

        metrics['max_r'] = np.mean(max_r).round(4)*t
        metrics['mean_r'] = np.mean(mean_r).round(4)* t
        metrics['rmsd'] = np.mean(rmsd).round(4)* t

    # metrics['norm'] = np.mean(norm).round(4)
    # metrics['norm_1'] = np.mean(norm_1).round(4)
    #
    # metrics['max_r'] = np.mean(max_r).round(4)
    # metrics['mean_r'] = np.mean(mean_r).round(4)
    # metrics['rmsd'] = np.mean(rmsd).round(4)
    print()
    print(metrics)
    print('can we here?')
    return metrics


#########################################
def testresult(model, ori_folder, output_folder, y_target, adv_num):
    print("Load Data from ori_folder")
    X = []
    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
    ])
    ori_imge = os.listdir(ori_folder)

    ori_imge.sort(key=lambda x: int(x[-7:-4]))
    # coco
    # ori_imge.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # ori_imge.sort(key=lambda x: int(x[0:6]))
    # ori_imge.sort(key=lambda x: int(x[0:-8]))

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
    y_target = y_target[0:adv_num, :]
    print(y_target.shape)
    state = {'model': model,
             'ori_folder': ori_folder,
             'output_folder': output_folder,
             'y_target': y_target,
             'num': adv_num
             }

    return evaluate_adv(state)
    # pre_adv(state)




if __name__ == "__main__":
    args = get_args()
    ###############################
    output_folder = args.output_folder
    ori_folder = args.ori_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    use_gpu = torch.cuda.is_available()
    torch.manual_seed(1)
    if use_gpu:
        torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    ###############################

    m_ty = "mlgcn"
    modelname = 'gcn2012'
    attackt = "hideall"
    datasetn = "voc2012"
    # inipath = './hidesingle/GCN_2007_adv/'

    qnummmm = 30000

    model = get_model(modelname).cuda()

    npy_path = './npy/{}/{}/'.format(modelname, attackt)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    adv_save_path = './adv_save/{}/{}/'.format(modelname, attackt)
    if not os.path.exists(adv_save_path):
        os.makedirs(adv_save_path)

    ori_img_path = './{}ori/'.format(datasetn)
    # ori_img_path = './hidesingle/ori_2012gcn/'

    ori_448_path = './{}ori448/'.format(datasetn)
    if not os.path.exists(ori_448_path):
        os.makedirs(ori_448_path)

    # adv_file_list = os.listdir(adv_save_path)
    #
    # adv_file_list.sort(key=lambda x: int(x[0:-8]))
    # coco
    # adv_file_list.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # adv_file_list.sort(key=lambda x: int(x[0:6]))

    # ini_list = os.listdir(inipath)

    # adv_file_list.sort(key=lambda x: int(x[0:-8]))
    # coco
    # adv_file_list.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # adv_file_list.sort(key=lambda x: int(x[0:6]))

    ############# ATTACK TYPE ###############################
    gen_adv_file(model, 'hide_all', ori_img_path, npy_path)
    ###   'hide_single'  'hide_all'     'random_case'   'extreme_case'   'person_reduction'  'sheep_augmentation'

    ###############   DATA   ###################

    y_t = np.load(os.path.join(npy_path, 'y_target.npy'))

    testresult(model, ori_img_path, os.path.join(adv_save_path, '3000/'), y_t,60)

























    #
    #
    # #  'gcn2007'  'gcn2012'   'liw2007'  'liw2012'
    # print("Load Model",args.modelname)
    # model = get_model(args.modelname)
    # ###############################
    # print("Load Config")
    #
    # config0 = json.load(open(args.config_path0, "r"))
    # config1 = json.load(open(args.config_path1, "r"))
    # config2 = json.load(open(args.config_path2, "r"))
    # # if args.config_path is not None:
    # #     if not os.path.exists(args.config_path):
    # #         raise ValueError("{} doesn't exist.".format(args.config_path))
    # #     config = json.load(open(args.config_path, "r"))
    # #
    # #
    # # else:
    # #     config = {"init": {}, "run": {"epsilons": None}}
    #
    # ###############################
    #
    # imagenet_labels = {'0': 'aeroplane','1': 'bicycle','2': 'bird','3': 'boat','4': 'bottle','5': 'bus','6': 'car','7': 'cat','8': 'chair','9': 'cow',
    #        '10': 'diningtable','11': 'dog','12': 'horse','13': 'motorbike','14': 'person','15': 'pottedplant','16': 'sheep','17': 'sofa','18': 'train','19': 'tvmonitor',}
    #
    # ###############################
    # ori_img_path = ''
    #
    #
    # print("Load Data")
    #
    #
    # transform = transforms.Compose([
    #     transforms.Resize((448, 448)),  # 设置目标尺寸为 100x100
    #     transforms.ToTensor()  # 转换为张量
    # ])
    # # transform = T.Compose([T.Resize(448), T.CenterCrop(448)])
    #
    # ori_imge = os.listdir(ori_folder)
    #
    # # #voc
    # # ori_imge.sort(key=lambda x: int(x[-7:-4]))
    # # #coco
    # # ori_imge.sort(key=lambda x: int(x[-10:-4]))
    # # nuswide
    # ori_imge.sort(key=lambda x: int(x[0:6]))
    #
    # X = []
    # cou = 0
    #
    # with torch.no_grad():
    #     for img in ori_imge:
    #         if cou > 20:
    #             break
    #         cou+=1
    #
    #         X = transforms.Compose([transforms.Resize((448, 448)),transforms.ToTensor() ])(Image.open(os.path.join(ori_folder, img))).unsqueeze(0)
    #         if torch.cuda.is_available():
    #             model = model.cuda()
    #             X = X.cuda()
    #             # y = y.cuda()
    #
    #
    #     # X = torch.cat(X, 0)
    #         print(X.shape)
    #         y = model(X)
    #         y1 = torch.where((model(X)-0.5)>0,1,0)
    #
    #
    #
    #
    #
    #     #######################    ini single
    #     # Ini = []
    #     #
    #     # ini_imge = os.listdir(args.ini_path)
    #     # ini_imge.sort(key=lambda x: int(x[0:-8]))
    #     #
    #     # cou = 0
    #     # with torch.no_grad():
    #     #     for img in ini_imge:
    #     #         # if cou > 3:
    #     #         #     break
    #     #         # cou+=1
    #     #         # print(os.path.join(args.ini_path, img))
    #     #         Ini.append(transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    #     #                  (Image.open(os.path.join(args.ini_path, img))).unsqueeze(0))
    #     #
    #     #     Ini = torch.cat(Ini, 0)
    #     #     print(Ini.shape)
    #     #
    #     #     y_ini = model(Ini)
    #     #     y_ini1 = torch.where((model(Ini) - 0.5) > 0, 1, 0)
    #     #
    #     #
    #     # print('shi fou pipei',X.shape)
    #     # print(Ini.shape)
    #     #
    #     # # print(y_ini1)
    #     # y_target = y_ini1
    #
    #     # print('shi fou pipei',y_target.shape)
    #     # print(torch.zeros_like(y1).shape)
    #
    #         y_target = torch.zeros_like(y1)
    #
    #         ###############################
    #         print("Attack !")
    #         time_start = time.time()
    #
    #
    #
    #
    #         for num in [300,3000,30000]:
    #
    #             if num == 300:
    #                 f_attack = SurFree(**config0["init"])
    #                 advs = f_attack(model, X, y,y_target, **config0["run"])
    #             elif num ==3000:
    #                 f_attack = SurFree(**config1["init"])
    #                 advs = f_attack(model, X, y,y_target, **config1["run"])
    #             elif num ==30000:
    #                 f_attack = SurFree(**config2["init"])
    #                 advs = f_attack(model, X, y, y_target,**config2["run"])
    #             #######################有目标###################
    #             # advs = f_attack(model, X, y, y2,X_tar, **config["run"])
    #
    #             ###############################
    #             #
    #             print("Results")
    #             labels_advs = model(advs)
    #             nqueries = f_attack.get_nqueries()
    #             advs_l2 = (X - advs).norm(dim=[1, 2]).norm(dim=1)
    #             print(type(advs_l2))
    #             for image_i in range(len(X)):
    #                 print("Adversarial Image {}:".format(image_i))
    #                 # print(y[image_i])
    #
    #                 label_o = ((y[image_i]>0.5)+0)
    #                 label_adv = ((labels_advs[image_i]>0.5)+0)
    #                 print(label_o)
    #                 print(label_adv)
    #                 print("\t- l2 = {}".format(advs_l2[image_i]))
    #                 print("\t- {} queries\n".format(nqueries[image_i]))
    #
    #             ###############################
    #             print("Save Results")
    #             for image_i, o in enumerate(X):
    #
    #
    #                 ###### 张量存储
    #                 # o = o.cpu().numpy()
    #                 # o = np.array(o).astype(np.uint8)
    #                 # # print(o.shape)
    #                 # np.save(ori_folder + "{}_ori.npy".format(image_i), o)
    #                 #
    #                 # adv_i = np.array(advs[image_i].cpu().numpy()).astype(np.uint8)
    #                 # print('各自最大值',np.max(o),np.max(adv_i))
    #                 # advs_l2= np.linalg.norm((adv_i-o).flatten())
    #                 # print("2222222222222222222222222222范数",advs_l2)
    #                 # # print(adv_i.shape)
    #                 # # print('tuxiangde chicun',adv_i.shape)
    #                 # np.save(output_folder + "{}_adv.npy".format(image_i), adv_i)
    #
    #                 ###### 图像存储
    #                 o = o.cpu().numpy()
    #                 o = np.array(o * 255).astype(np.uint8)
    #                 img_o = Image.fromarray(o.transpose(1, 2, 0), mode="RGB")
    #                 if not os.path.exists(args.ori_folder_save):
    #                     os.makedirs(args.ori_folder_save)
    #                 img_o.save(os.path.join(args.ori_folder_save, "{}_ori.png".format(image_i)))
    #
    #                 adv_i = np.array(advs[image_i].cpu().numpy() * 255).astype(np.uint8)
    #                 img_adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB")
    #                 if not os.path.exists(os.path.join(args.output_folder, args.attacktype,'{}'.format(num))):
    #                     os.makedirs(os.path.join(args.output_folder, args.attacktype,'{}'.format(num)))
    #
    #                 img_adv_i.save(os.path.join(args.output_folder, args.attacktype,'{}'.format(num),'{}_adv.png'.format(cou-1)))
    #
    #             # break














