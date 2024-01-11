import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

print(torch.cuda.device_count())
print(torch.cuda.device_count())
import sys
sys.path.append('../')
import pandas as pd
import argparse
import numpy as np
import logging
from tqdm import tqdm
import torchvision.transforms as transforms
from ml_gcn_model.models import gcn_resnet101_attack
from ml_gcn_model.voc import Voc2007Classification
from ml_gcn_model.voc import Voc2012Classification
from ml_gcn_model.util import Warp
from ml_gcn_model.voc import write_object_labels_csv
from src.attack_model_2012 import AttackModel

parser = argparse.ArgumentParser(description='multi-label attack')
parser.add_argument('--data', default='../data/voc2012', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--image_size', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 100)')
parser.add_argument('--adv_batch_size', default=1, type=int,
                    metavar='N', help='batch size ml_cw, ml_rank1, ml_rank2 25, ml_lp 5, ml_deepfool is 10')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--adv_method', default='ml_ba', type=str, metavar='N',
                    help='attack method: ml_cw, ml_rank1, ml_rank2, ml_deepfool, ml_de5')
parser.add_argument('--target_type', default='hide_all', type=str, metavar='N',
                    help='target method: hide_all、hide_single')
parser.add_argument('--adv_file_path', default='../data/voc2012/files/VOC2012/classification_mlgcn_adv_0_99_cut.csv', type=str, metavar='N',
                    help='all image names and their labels ready to attack')
parser.add_argument('--adv_save_x', default='../adv_save/mlgcn/voc2012/0_99_cut', type=str, metavar='N',
                    help='save adversiral examples')
parser.add_argument('--adv_begin_step', default=34, type=int, metavar='N',
                    help='which step to start attacking according to the batch size')


parser.add_argument('--save_image_folder', default="GCN_2012", type=str, metavar='N',
                    help='which step to start attacking according to the batch size')

def new_folder(file_path):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def init_log(log_file):
  new_folder(log_file)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

  fh = logging.FileHandler(log_file)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)

  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  ch.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)

def get_target_label(y, target_type):
    '''
    :param y: numpy, y in {0, 1}
    :param A: list, label index that we want to reverse
    :param C: list, label index that we don't care
    :return:
    '''
    y = y.copy()
    # o to -1
    y[y == 0] = -1
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
        y[:, 14] = -y[:, 14]
    elif target_type == 'sheep_augmentation':
        # sheep in 17 col
        y[:, 17] = -y[:, 17]
    elif target_type == 'hide_single':
        for i, y_i in enumerate(y):
            pos_idx = np.argwhere(y_i == 1).flatten()
            pos_idx_c = np.random.choice(pos_idx)
            y[i, pos_idx_c] = -y[i, pos_idx_c]
    elif target_type == 'hide_all':
            y[y == 1] = -1
    return y

def gen_adv_file(model, target_type, adv_file_path):
    print("generiting……")
    tqdm.monitor_interval = 0
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    test_dataset = Voc2012Classification(args.data, 'val', inp_name='../data/voc2012/voc_glove_word2vec.pkl')
    test_dataset.transform = data_transforms
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    output = []
    image_name_list = []
    y = []
    test_loader = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            x = input[0]
            if use_gpu:
                x = x.cuda()
            o = model(x).cpu().numpy()
            output.extend(o)
            y.extend(target.cpu().numpy())
            image_name_list.extend(list(input[1]))
        output = np.asarray(output)
        y = np.asarray(y)
        image_name_list = np.asarray(image_name_list)

    # choose x which can be well classified and contains two or more label to prepare attack
    pred = (output >= 0.5) + 0
    y[y==-1] = 0
    true_idx_tem = []
    true_idx = []
    idx_tongji = []
    count = 0
    for i in range(len(pred)):
        if (y[i] == pred[i]).all() and np.sum(y[i]) >= 2 and count>=0 and count<=49:
            true_idx.append(i)   #有if时与if冲齐
            idx_tongji.append(i+1) #有if时与if冲齐
            count += 1
        # if (y[i] == pred[i]).all() and count>=0 and count<=99:
        #     true_idx.append(i)   #有if时与if冲齐
        #     idx_tongji.append(i+1) #有if时与if冲齐
        #     count += 1
    dataframe = pd.DataFrame(idx_tongji)
    dataframe.to_excel('攻击列表.xls')
    adv_image_name_list = image_name_list[true_idx]
    adv_y = y[true_idx]
    y = y[true_idx]
    y_target = get_target_label(adv_y, target_type)
    y_target[y_target==0] = -1
    y[y==0] = -1
    print(len(adv_image_name_list))

    adv_labeled_data = {}
    for i in range(len(adv_image_name_list)):
        adv_labeled_data[adv_image_name_list[i]] = y[i]
    write_object_labels_csv(adv_file_path, adv_labeled_data)

    # save target y and ground-truth y to prepare attack
    # value is {-1,1}
    np.save('../adv_save/mlgcn/voc2012/y_target_0_99_cut.npy', y_target)
    np.save('../adv_save/mlgcn/voc2012/y_0_99_cut.npy', y)



def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()


    # set seed
    torch.manual_seed(123)
    if use_gpu:
        torch.cuda.manual_seed_all(123)
    np.random.seed(123)

    init_log(os.path.join(args.adv_save_x, args.adv_method, args.target_type + '_yolo.log'))

    # define dataset
    num_classes = 20

    # load torch model
    model = gcn_resnet101_attack(num_classes=num_classes,
                                 t=0.4,
                                 adj_file='../data/voc2012/voc_adj.pkl',
                                 word_vec_file='../data/voc2012/voc_glove_word2vec.pkl',
                                 save_model_path='../checkpoint/mlgcn/voc2012/voc_checkpoint.pth.tar')

    model.eval()
    if use_gpu :
        model = model.cuda()
    if not os.path.exists(args.adv_file_path):
       gen_adv_file(model, args.target_type, args.adv_file_path)
    # gen_adv_file(model, args.target_type, args.adv_file_path)
    # #
    # transfor image to torch tensor
    # the tensor size is [chnnel, height, width]
    # the tensor value in [0,1]
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    ori_dataset = Voc2007Classification("../data/voc2007", 'test', inp_name='../data/voc2007/voc_glove_word2vec.pkl')
    ori_dataset.transform = data_transforms
    ori_loader = torch.utils.data.DataLoader(ori_dataset,
                                             batch_size=args.adv_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers)

    adv_dataset = Voc2012Classification(args.data, 'mlgcn_adv_0_99_cut',
                                        inp_name='../data/voc2012/voc_glove_word2vec.pkl')
    adv_dataset.transform = data_transforms
    adv_loader = torch.utils.data.DataLoader(adv_dataset,
                                              batch_size=args.adv_batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)

    # load target y and ground-truth y
    # value is {-1,1}
    y_target = np.load('../adv_save/mlgcn/voc2012/y_target_0_99_cut.npy')
    y = np.load('../adv_save/mlgcn/voc2012/y_0_99_cut.npy')

    state = {'model': model,
             'data_loader': adv_loader,
             'ori_loader': ori_loader,
             'adv_method': args.adv_method,
             'target_type': args.target_type,
             'adv_batch_size': args.adv_batch_size,
             'y_target':y_target,
             'y': y,
             'adv_save_x': os.path.join(args.adv_save_x, args.adv_method, args.target_type + '.npy'),
             'adv_begin_step': args.adv_begin_step,
             'save_image_folder':args.save_image_folder
             }

    # start attack
    attack_model = AttackModel(state)
    attack_model.attack()
    # evaluate_adv(state)
    #evaluate_model(model)

if __name__ == '__main__':
    main()