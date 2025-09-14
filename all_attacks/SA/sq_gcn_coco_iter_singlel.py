import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torchvision import transforms
import argparse
import time
import numpy as np
import scipy.misc
import data
import tqdm
import models
import utils
from datetime import datetime
from PIL import Image

from ml_gcn_model.models import gcn_resnet101_attack
from ml_gcn_model.voc import Voc2007Classification
from ml_gcn_model.util import Warp
import torchvision.transforms as transforms
from ml_gcn_model.voc import write_object_labels_csv

from ml_liw_model.models import inceptionv3_attack
from ml_liw_model.voc import Voc2012Classification
from ml_gcn_model.util import Warp
from ml_liw_model.voc import write_object_labels_csv

np.set_printoptions(precision=5, suppress=True)


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
    samplenum = state['num']
    y_target = state['y_target']
    adv_folder_path = state['output_folder']
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x: int(x[0:-8]))
    ori_folder_path = state['ori_folder']
    ori_file_list = os.listdir(ori_folder_path)

    ori_file_list.sort(key=lambda x: int(x[-7:-4]))
    #coco
    # ori_file_list.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # ori_file_list.sort(key=lambda x: int(x[0:6]))



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
    adv_pred_match_target = (np.logical_and(
        np.logical_and(np.all((adv_pred == y_target), axis=1),
                       np.asarray(norm) < (77.6 *2))  ,np.asarray(norm) >(0.00)) ) + 0
    # adv_pred_match_target = (np.asarray(norm) < (77.6 * 1)) + 0

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

    # t =1.15
    # for i in ['norm','norm_1','max_r','mean_r','rmsd']:
    #     metrics['norm'] = np.mean(norm).round(4)  * t
    #     metrics['norm_1'] = np.mean(norm_1).round(4)* t
    #
    #     metrics['max_r'] = np.mean(max_r).round(4)*t
    #     metrics['mean_r'] = np.mean(mean_r).round(4)* t
    #     metrics['rmsd'] = np.mean(rmsd).round(4)* t


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
    #coco
    # ori_imge.sort(key=lambda x: int(x[-10:-4]))
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

################### attack Function  ###########################
def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p

def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
        max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta

def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def get_loss(model, x, y_tar):
    output = model(torch.tensor(x).float().cuda()).cpu().detach().numpy()
    pred_x = (output>0.5)+0
    p = np.copy(np.asarray(output))
    q = np.zeros(p.shape) + 0.5
    fit = p-q
    # print(fit)
    fit[:, y_tar] = -fit[:, y_tar]
    fit[np.where(fit < 0)] = 0
    loss = np.sum(fit, axis=1)
    # print('dang qian loss',loss)
    return loss,pred_x[0]

def square_attack_linf(model, x, y_target, eps, n_iters, p_init):
    print('e={}'.format(eps),'iter={}'.format(n_iters))


    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well

    # print('target is:',y_target)

    y_tar = np.argwhere(y_target.squeeze() != 0).flatten()

    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    # w, h, c = x.shape[1:]
    n_features = c * h * w
    n_ex_total = x.shape[0]
    # x, y = x[corr_classified], y[corr_classified]  #shifou bei zheng que fen lei

    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    # init_delta111 = np.random.choice([-eps, eps], size=[x.shape[0], 1, w, c])
    # print(init_delta111.shape)
    x_best = np.clip(x + init_delta, min_val, max_val)

    loss_min, _ = get_loss(model, x_best, y_tar)


    # logits = model.get_prob_(x_best)
    # loss_min = model.loss(y_target, logits)
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    for i_iter in range(n_iters - 1):
        idx_to_fool = [0]
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y_target
        loss_min_curr = loss_min[idx_to_fool]
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h + s, center_w:center_w + s],
                                        min_val, max_val) - x_best_curr_window) < 10 ** -7) == c * s * s:
                deltas[i_img, :, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-eps, eps],
                                                                                                  size=[c, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        loss , pred_label= get_loss(model, x_new, y_tar)

        if loss ==0:
            print('sucess {}'.format(loss))
            return n_queries, x_new


        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        sucess = get_decision(pred_label, y_curr)

        if sucess:
            print('sucess {}'.format(n_iters))
            # print('y final is:', y_target)
            return n_queries, x_best
    return n_queries, x_best

def get_decision(prob, y_target):
    # print(prob == y_target)
    # print((prob == y_target).all())
    if (prob == y_target).all():
        return True
    return False

def gen_adv_file(model, target_type, ori_img_path,npy_path):
    print("generiting target file…")

    advlist = os.listdir(ori_img_path)

    advlist.sort(key=lambda x: int(x[-7:-4]))
    #coco
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
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    # parser.add_argument('--model', type=str, default='pt_resnet', choices=models.all_model_names, help='Model name.')
    parser.add_argument('--attack', type=str, default='square_linf', choices=['square_linf', 'square_l2'],
                        help='Attack.')
    parser.add_argument('--n_iter', type=int, default=30000)
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.')
    parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
    parser.add_argument('--targeted', action='store_true', help='Targeted or untargeted attack.')
    args = parser.parse_args()

    batch_size = 1
    n_cls = 20

    torch.manual_seed(998)
    torch.cuda.manual_seed(998)
    np.random.seed(998)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###############   MODEL    ###################

    m_ty = "mlgcn"
    modelname = 'gcn2007'
    attackt = "random_case"
    datasetn = "voc2007"

    qnummmm = 30000

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

    adv_file_list.sort(key=lambda x: int(x[-7:-4]))
    #coco
    # adv_file_list.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # adv_file_list.sort(key=lambda x: int(x[0:6]))


    ############# ATTACK TYPE ###############################
    gen_adv_file(model, 'random_case', ori_img_path, npy_path)
    ###   'hide_single'  'hide_all'     'random_case'   'extreme_case'   'person_reduction'  'sheep_augmentation'



    ##############  ATTACKER  ###################
    square_attack = square_attack_linf
    ###############   DATA   ###################




    y_t = np.load(os.path.join(npy_path, 'y_target.npy'))


    # testresult(model, ori_img_path, os.path.join(adv_save_path, '300/'), y_t, 50)


    # #
    for num in [3000]:
        print('NOW  ITER = 300')
        print(num)
        ii = 1
        for f in adv_file_list:
            if ii >50:
                break
            # print(y_target)
            test = Image.open(os.path.join(ori_img_path, f)).resize((448, 448)).convert('RGB')
            test.save(os.path.join(ori_448_path, '{m}.png'.format(m=ii)))
            print('This is num {} pic\n'.format(ii))

            # print(os.path.join(adv_folder_path,f))

            x_test = (np.array(test).transpose(2, 0, 1).astype('float32') / 255.).reshape(1, 3, 448, 448)

            # yooo = model(torch.tensor(x_test).float().cuda()).cpu().detach().numpy()
            # yooo =  (yooo>0.5)+0
            # print('y ori is:',yooo)

            y_target = y_t[ii-1]

            print('this tar:' ,y_target)

            n_queries, x_adv = square_attack(model, x_test, y_target, args.eps, num,
                                             args.p)


            print((model(torch.tensor(x_adv).float().cuda()).cpu().detach().numpy()>0.5)+0)

            img = Image.fromarray(np.uint8(x_adv.squeeze() * 255).transpose(1, 2, 0), 'RGB')

            if not os.path.exists(os.path.join(adv_save_path, '{n}/'.format(n=num))):
                os.makedirs(os.path.join(adv_save_path, '{n}/'.format(n=num)))

            img.save(os.path.join(adv_save_path, '{n}/{m}_adv.png'.format(n=num,m=ii)))

            ii += 1
    #
