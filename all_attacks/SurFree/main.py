import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'

import torch
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
from surfree_single import SurFree

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
    elif x == 'liw2007':
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2007/model_best.pth.tar')
    elif x == 'liw2012':
        model = inceptionv3_attack(num_classes=num_classes,
                                   save_model_path='./checkpoint/mlliw/voc2012/model_best.pth.tar')

    return model.eval()


    # model = torchvision.models.resnet18(pretrained=True).eval()
    # mean = torch.Tensor([0.485, 0.456, 0.406])
    # std = torch.Tensor([0.229, 0.224, 0.225])
    # normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
    # return torch.nn.Sequential(normalizer, model).eval()

def get_imagenet_labels():
    response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
    return eval(response.content)

def get_image_labels():
    response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
    return eval(response.content)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-out", default="results_test/", help="Output folder")
    parser.add_argument("--ori_folder", "-ori", default="ori/", help="Output folder")
    parser.add_argument("--n_images", "-n", type=int, default=2, help="N images attacks")
    parser.add_argument(
        "--config_path", 
        default="config_example.json", 
        help="Configuration Path with all the parameter for SurFree. It have to be a dict with the keys init and run."
        )
    return parser.parse_args()


def pre_adv(state):
    model = state['model']
    adv_folder_path = state['output_folder']
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x: int(x[0:-8]))

    ori_folder_path = state['ori_folder']
    ori_file_list = os.listdir(ori_folder_path)
    ori_file_list.sort(key=lambda x: int(x[0:-8]))

    adv = []
    for f in adv_file_list:
        # 读取图像
        a = np.asarray(Image.open(adv_folder_path + f)) / 255.
        # print(a.shape)

        # 读取数组
        # a = np.load(adv_folder_path + f, allow_pickle=True)

        # print('duqude chicun',np.expand_dims(a, axis=0).shape)
        adv.extend(np.expand_dims(a, axis=0))
    adv = np.asarray(adv)

    new_adv = []

    for img in adv:
        # print(img.shape)

        # img = np.transpose(img, (1, 2, 0))

        img = Image.fromarray(np.uint8(img * 255))
        crop = transforms.Resize(448)
        img = crop(img)
        img = np.asarray(img)
        img = img.astype('float32')
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        new_adv.append(img)
    new_adv = np.asarray(new_adv)
    # print(adv.dtype)
    # print(new_adv.dtype)

    adv = new_adv

    ori = []
    for f in ori_file_list:
        # 读取图像
        a = np.asarray(Image.open(ori_folder_path + f)) / 255.

        # 读取数组
        # a = np.load(ori_folder_path + f, allow_pickle=True)

        ori.extend(np.expand_dims(a, axis=0))
    ori = np.asarray(ori)
    new_ori = []
    for img in ori:
        # img = np.transpose(img, (1, 2, 0))

        img = Image.fromarray(np.uint8(img * 255))
        crop = transforms.Resize(448)
        img = crop(img)
        img = np.asarray(img)
        img = img.astype('float32')
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        new_ori.append(img)
    new_ori = np.asarray(new_ori)
    # print(adv.dtype)
    # print(new_adv.dtype)

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
    ori_output =  []

    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            if use_gpu:
                batch_adv_x = batch_adv_x.cuda()
                batch_test_x = batch_test_x.cuda()
            adv_output.extend(model(batch_adv_x).cpu().numpy())
            ori_output.extend(model(batch_test_x).cpu().numpy())
    adv_output = np.asarray(adv_output)
    ori_output = np.asarray(ori_output)
    adv_pred = adv_output.copy()
    ori_pred = ori_output.copy()
    adv_pred[adv_pred >= (0.5 + 0)] = 1
    adv_pred[adv_pred < (0.5 + 0)] = 0
    ori_pred[ori_pred >= (0.5 + 0)] = 1
    ori_pred[ori_pred < (0.5 + 0)] = 0
    print('原始图像的预测标签：\n',ori_pred)
    print('对抗样本的预测标签：\n',adv_pred)



def evaluate_adv(state):
    model = state['model']
    y_target = state['y_target']

    adv_folder_path = state['output_folder']
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x: int(x[0:-8]))

    ori_folder_path = state['ori_folder']
    ori_file_list = os.listdir(ori_folder_path)
    ori_file_list.sort(key=lambda x: int(x[0:-8]))




    # if state['target_type'] == 'hide_all':
    #     adv_file_list.sort(key=lambda x: int(x[13:-4]))
    # elif state['target_type'] == 'hide_single':
    #     adv_file_list.sort(key=lambda x: int(x[16:-4]))
    # elif state['target_type'] == 'random_case':
    #     adv_file_list.sort(key=lambda x: int(x[16:-4]))
    # elif state['target_type'] == 'extreme_case':
    #     adv_file_list.sort(key=lambda x: int(x[17:-4]))
    # elif state['target_type'] == 'person_reduction':
    #     adv_file_list.sort(key=lambda x: int(x[21:-4]))
    # elif state['target_type'] == 'sheep_augmentation':
    #     adv_file_list.sort(key=lambda x: int(x[23:-4]))
    # elif state['target_type'] == 'specific':
    #     adv_file_list.sort(key=lambda x: int(x[13:-4]))
    # elif state['target_type'] == 'open_all':
    #     adv_file_list.sort(key=lambda x: int(x[13:-4]))
    # else:
    #     print("wrong type index")
    # all-13;single-16;extream-17;random-16;person-21;sheep-23
    adv = []
    for f in adv_file_list:
        # 读取图像
        a = np.asarray(Image.open(adv_folder_path + f))/255.
        # print(a.shape)

        # 读取数组
        # a = np.load(adv_folder_path + f, allow_pickle=True)

        # print('duqude chicun',np.expand_dims(a, axis=0).shape)
        adv.extend(np.expand_dims(a, axis=0))
    adv = np.asarray(adv)

    new_adv = []

    for img in adv:
        # print(img.shape)

        # img = np.transpose(img, (1, 2, 0))

        img = Image.fromarray(np.uint8(img * 255))
        crop = transforms.Resize(448)
        img = crop(img)
        img = np.asarray(img)
        img = img.astype('float32')
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        new_adv.append(img)
    new_adv = np.asarray(new_adv)
    # print(adv.dtype)
    # print(new_adv.dtype)

    adv = new_adv

    ori = []
    for f in ori_file_list:
        # 读取图像
        a = np.asarray(Image.open(ori_folder_path + f))/255.


        # 读取数组
        # a = np.load(ori_folder_path + f, allow_pickle=True)

        ori.extend(np.expand_dims(a, axis=0))
    ori = np.asarray(ori)
    new_ori = []
    for img in ori:

        # img = np.transpose(img, (1, 2, 0))

        img = Image.fromarray(np.uint8(img * 255))
        crop = transforms.Resize(448)
        img = crop(img)
        img = np.asarray(img)
        img = img.astype('float32')
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        new_ori.append(img)
    new_ori = np.asarray(new_ori)
    # print(adv.dtype)
    # print(new_adv.dtype)

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
            if use_gpu:
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
    adv_pred[adv_pred >= (0.5+0)] = 1
    adv_pred[adv_pred < (0.5+0)] = -1
    # print(adv_pred.shape)
    # print(adv_pred)
    # print(y_target.shape)
    # print(y_target)

    # 原始 # 按照2范数来选择
    adv_pred_match_target = (np.logical_and(np.all((adv_pred == y_target), axis=1), np.asarray(norm)<(77.6*3)))+0
    # adv_pred_match_target = (np.asarray(norm) < (77.6 * 3)) + 0





    # print(adv_pred_match_target.shape)
    # print('dfsdfssgsdg',adv_pred_match_target)


    # print(np.all((adv_pred == y_target), axis=1))


    attack_fail_idx = np.argwhere(adv_pred_match_target==0).flatten()


    np.save('attack_fail_idx.npy', attack_fail_idx)
    norm = np.asarray(norm)

    print(norm)

    max_r = np.asarray(max_r).round(4)
    mean_r = np.asarray(mean_r).round(4)
    rmsd = np.asarray(rmsd).round(4)
    norm = np.delete(norm, attack_fail_idx, axis=0).round(4)
    max_r = np.delete(max_r, attack_fail_idx, axis=0).round(4)
    norm_1 = np.delete(norm_1, attack_fail_idx, axis=0).round(4)
    mean_r = np.delete(mean_r, attack_fail_idx, axis=0).round(4)
    rmsd = np.delete(rmsd, attack_fail_idx, axis=0).round(4)

    from utils import evaluate_metrics
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




if __name__ == "__main__":
    args = get_args()
    ###############################
    output_folder = args.output_folder
    ori_folder = args.ori_folder
    if not os.path.exists(output_folder):
        raise ValueError("{} doesn't exist.".format(output_folder))

    use_gpu = torch.cuda.is_available()
    torch.manual_seed(123)
    if use_gpu:
        torch.cuda.manual_seed_all(123)
    np.random.seed(123)

    ###############################
    #  'gcn2007'  'gcn2012'   'liw2007'  'liw2012'
    modelname = 'gcn2007'
    print("Load Model",modelname)
    model = get_model(modelname)


    ###############################
    print("Load Config")
    if args.config_path is not None:
        if not os.path.exists(args.config_path):
            raise ValueError("{} doesn't exist.".format(args.config_path))
        config = json.load(open(args.config_path, "r"))
    else:
        config = {"init": {}, "run": {"epsilons": None}}

    ###############################
    # print("Get VOC labels")
    # imagenet_labels = get_imagenet_labels()

    imagenet_labels = {'0': 'aeroplane','1': 'bicycle','2': 'bird','3': 'boat','4': 'bottle','5': 'bus','6': 'car','7': 'cat','8': 'chair','9': 'cow',
           '10': 'diningtable','11': 'dog','12': 'horse','13': 'motorbike','14': 'person','15': 'pottedplant','16': 'sheep','17': 'sofa','18': 'train','19': 'tvmonitor',}

    # print(type(voc_labels))
    # print(voc_labels)
    # print(type(imagenet_labels))
    
    ###############################
    print("Load Data")
    X = []
    transform = T.Compose([T.Resize(448), T.CenterCrop(448)])

    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
    ])

    ori_imge = os.listdir("./images/JPEGImages")
    ori_imge.sort(key=lambda x: int(x[-7:-4]))

    for img in ori_imge:
        a = Image.open(os.path.join("./images/JPEGImages", img))
        a= data_transforms(a).unsqueeze(0)
        # print(a.shape)
        X.append(a)
        # X.append(transform(read_image(os.path.join("./images/JPEGImages", img))).unsqueeze(0))

    X = torch.cat(X, 0)
    # X = torch.cat(X, 0) / 255
    y = model(X).argmax(1)

    y1 = torch.where((model(X)-0.5)>0,1,0)
    # print(y1)
    # print(y1.shape)
    # print(type(y1))


    #######################    hide all
    y2 = torch.zeros_like(y1)
    # print(y2.shape)
    # print(y2)


    #######################    hide sing
    ################目标图像##############
    # X_tar = []
    # transform = T.Compose([T.Resize(448), T.CenterCrop(448)])
    # for img in os.listdir("./images/tarimg"):
    #     X_tar.append(transform(read_image(os.path.join("./images/JPEGImages", img))).unsqueeze(0))
    # X_tar = torch.cat(X_tar, 0) / 255
    # y = model(X_tar)


    ###############################
    print("Attack !")
    time_start = time.time()

    f_attack = SurFree(**config["init"])

    if torch.cuda.is_available():
        model = model.cuda()
        X = X.cuda()
        y = y.cuda()

    print(X.shape)
    print(y)

    advs = f_attack(model, X, y,y2, **config["run"])

    #######################有目标###################
    # advs = f_attack(model, X, y, y2,X_tar, **config["run"])

    ###############################


    print("Results")
    labels_advs = model(advs).argmax(1)
    nqueries = f_attack.get_nqueries()
    advs_l2 = (X - advs).norm(dim=[1, 2]).norm(dim=1)
    print(type(advs_l2))
    for image_i in range(len(X)):
        print("Adversarial Image {}:".format(image_i))
        label_o = int(y[image_i])
        # print("image_i\n",image_i)
        # print("label_o\n",label_o)
        label_adv = int(labels_advs[image_i])
        # print("labels_advs\n",labels_advs)
        # print("label_adv\n",label_adv)
        # print("\t- Original label: {}".format(imagenet_labels[str(label_o)]))
        # print("\t- Adversarial label: {}".format(imagenet_labels[str(label_adv)]))
        print("\t- l2 = {}".format(advs_l2[image_i]))
        print("\t- {} queries\n".format(nqueries[image_i]))

    ###############################
    print("Save Results")
    for image_i, o in enumerate(X):


        ###### 张量存储
        # o = o.cpu().numpy()
        # o = np.array(o).astype(np.uint8)
        # # print(o.shape)
        # np.save(ori_folder + "{}_ori.npy".format(image_i), o)
        #
        # adv_i = np.array(advs[image_i].cpu().numpy()).astype(np.uint8)
        # print('各自最大值',np.max(o),np.max(adv_i))
        # advs_l2= np.linalg.norm((adv_i-o).flatten())
        # print("2222222222222222222222222222范数",advs_l2)
        # # print(adv_i.shape)
        # # print('tuxiangde chicun',adv_i.shape)
        # np.save(output_folder + "{}_adv.npy".format(image_i), adv_i)


        ###### 图像存储
        o = o.cpu().numpy()
        o = np.array(o * 255).astype(np.uint8)
        img_o = Image.fromarray(o.transpose(1, 2, 0), mode="RGB")
        img_o.save(os.path.join(ori_folder, "{}_ori.png".format(image_i)))

        adv_i = np.array(advs[image_i].cpu().numpy() * 255).astype(np.uint8)
        img_adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB")
        img_adv_i.save(os.path.join(output_folder, "{}_adv.png".format(image_i)))

        # break


    # #####################    评估 评估 评估 评估    ##############################
    # print("evaluate !")
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     X = X.cuda()
    #     y = y.cuda()
    # y_target =  np.zeros((4,20))-1
    # # y_target = np.load('./y_t.npy')
    #
    # # print(y_target.shape)
    # state = {'model': model,
    #          'ori_folder': ori_folder,
    #          'output_folder': output_folder,
    #          'y_target': y_target
    #          }
    # evaluate_adv(state)
    # # pre_adv(state)
    # ###################################













