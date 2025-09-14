import os
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", "-model", default="liw2007", help="modelname")
    parser.add_argument("--attacktype", "-attack", default="hideall", help="hideall, hidesingle")
    parser.add_argument("--config_path0", default="config_300.json",help="Configuration")
    parser.add_argument("--config_path1", default="config_3000.json", help="Configuration")
    parser.add_argument("--config_path2", default="config_30000.json", help="Configuration")
    # parser.add_argument("--qnum", default="300",help="qnum")
## single##
    # parser.add_argument("--ini_path", "-ini", default="./hidesingle/LIW_2012_adv/", help="ini_path")
    parser.add_argument("--ori_folder", "-ori", default="./ori_2007/", help="ori folder")
    parser.add_argument("--ori_folder_save", "-ori_save", default="./ori_2007_448/", help="ori folder")

###
    # parser.add_argument("--ori_folder", "-ori", default="./adv_save/gcn2007/ori/", help="ori folder")

    parser.add_argument("--output_folder", "-out", default="./adv_save/liw2007/", help="Output folder")


    parser.add_argument("--n_images", "-n", type=int, default=2, help="N images attacks")

    return parser.parse_args()



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
    #  'gcn2007'  'gcn2012'   'liw2007'  'liw2012'
    print("Load Model",args.modelname)
    model = get_model(args.modelname)
    ###############################
    print("Load Config")

    config0 = json.load(open(args.config_path0, "r"))
    config1 = json.load(open(args.config_path1, "r"))
    config2 = json.load(open(args.config_path2, "r"))
    # if args.config_path is not None:
    #     if not os.path.exists(args.config_path):
    #         raise ValueError("{} doesn't exist.".format(args.config_path))
    #     config = json.load(open(args.config_path, "r"))
    #
    #
    # else:
    #     config = {"init": {}, "run": {"epsilons": None}}

    ###############################

    imagenet_labels = {'0': 'aeroplane','1': 'bicycle','2': 'bird','3': 'boat','4': 'bottle','5': 'bus','6': 'car','7': 'cat','8': 'chair','9': 'cow',
           '10': 'diningtable','11': 'dog','12': 'horse','13': 'motorbike','14': 'person','15': 'pottedplant','16': 'sheep','17': 'sofa','18': 'train','19': 'tvmonitor',}

    ###############################
    ori_img_path = ''


    print("Load Data")

    X = []
    transform = transforms.Compose([
        transforms.Resize((448, 448)),  # 设置目标尺寸为 100x100
        transforms.ToTensor()  # 转换为张量
    ])
    # transform = T.Compose([T.Resize(448), T.CenterCrop(448)])

    ori_imge = os.listdir(ori_folder)
    ori_imge.sort(key=lambda x: int(x[-7:-4]))

    cou = 0

    with torch.no_grad():
        for img in ori_imge:
            # if cou > 3:
            #     break
            # cou+=1

            X.append(transforms.Compose([transforms.Resize((448, 448)),transforms.ToTensor() ])
                     (Image.open(os.path.join(ori_folder, img))).unsqueeze(0))
        X = torch.cat(X, 0)
        # print(X.shape)
        y = model(X)
        y1 = torch.where((model(X)-0.5)>0,1,0)


    # #######################    ini single
    # Ini = []
    #
    # ini_imge = os.listdir(args.ini_path)
    # ini_imge.sort(key=lambda x: int(x[0:-8]))
    #
    # cou = 0
    # with torch.no_grad():
    #     for img in ini_imge:
    #         # if cou > 3:
    #         #     break
    #         # cou+=1
    #         # print(os.path.join(args.ini_path, img))
    #         Ini.append(transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    #                  (Image.open(os.path.join(args.ini_path, img))).unsqueeze(0))
    #
    #     Ini = torch.cat(Ini, 0)
    #     # print(Ini.shape)
    #
    #     y_ini = model(Ini)
    #     y_ini1 = torch.where((model(Ini) - 0.5) > 0, 1, 0)
    #
    # #
    # # print('shi fou pipei',X.shape)
    # # print(Ini.shape)
    #
    # # print(y_ini1)
    # y_target = y_ini1
    #
    # # print('shi fou pipei',y_target.shape)
    # # print(torch.zeros_like(y1).shape)

    y_target = torch.zeros_like(y1)

    ###############################
    print("Attack !")
    time_start = time.time()

    if torch.cuda.is_available():
        model = model.cuda()
        X = X.cuda()
        y = y.cuda()


    for num in [30000]:

        if num == 300:
            f_attack = SurFree(**config0["init"])
            advs = f_attack(model, X, y,y_target, **config0["run"])
        elif num ==3000:
            f_attack = SurFree(**config1["init"])
            advs = f_attack(model, X, y,y_target, **config1["run"])
        elif num ==30000:
            f_attack = SurFree(**config2["init"])
            advs = f_attack(model, X, y, y_target,  **config2["run"])
        #######################有目标###################
        # advs = f_attack(model, X, y, y2,X_tar, **config["run"])

        ###############################
        #
        print("Results")
        labels_advs = model(advs)
        nqueries = f_attack.get_nqueries()
        advs_l2 = (X - advs).norm(dim=[1, 2]).norm(dim=1)
        print(type(advs_l2))
        for image_i in range(len(X)):
            print("Adversarial Image {}:".format(image_i))
            # print(y[image_i])

            label_o = ((y[image_i]>0.5)+0)
            label_adv = ((labels_advs[image_i]>0.5)+0)
            print(label_o)
            print(label_adv)
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
            if not os.path.exists(args.ori_folder_save):
                os.makedirs(args.ori_folder_save)
            img_o.save(os.path.join(args.ori_folder_save, "{}_ori.png".format(image_i)))

            adv_i = np.array(advs[image_i].cpu().numpy() * 255).astype(np.uint8)
            img_adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB")
            if not os.path.exists(os.path.join(args.output_folder, args.attacktype,'{}'.format(num))):
                os.makedirs(os.path.join(args.output_folder, args.attacktype,'{}'.format(num)))

            img_adv_i.save(os.path.join(args.output_folder, args.attacktype,'{}'.format(num),'{}_adv.png'.format(image_i)))

        # break














