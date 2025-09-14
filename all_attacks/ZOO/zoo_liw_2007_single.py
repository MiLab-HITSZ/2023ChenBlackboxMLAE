import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torchvision import transforms

import time
import numpy as np

import imageio


from ml_gcn_model.models import gcn_resnet101_attack
from ml_gcn_model.voc import Voc2007Classification
from ml_gcn_model.util import Warp
import torchvision.transforms as transforms
from ml_gcn_model.voc import write_object_labels_csv

from ml_liw_model.models import inceptionv3_attack
from ml_liw_model.voc import Voc2012Classification
from ml_gcn_model.util import Warp
from ml_liw_model.voc import write_object_labels_csv

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms, datasets
from numba import jit
import math
import time
import scipy.misc
import os
import sys
from PIL import Image




"""##L2 Black Box Attack"""




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






def get_decision(x, y):
    x = (x>0.5)+0

    y = (y>0.5)+0

    if (x == y).all():
        return True
    return False



def get_loss( output, y_target):
    output=output.cpu().numpy()
    y_target = y_target.cpu().numpy()
    y_tar = np.argwhere(y_target.squeeze() != 0).flatten()
    pred_x = (output > 0.5) + 0
    p = np.copy(np.asarray(output))
    q = np.zeros(p.shape) + 0.5
    fit = p - q
    # print(fit)
    fit[:, y_tar] = -fit[:, y_tar]
    fit[np.where(fit < 0)] = 0
    loss = np.sum(fit, axis=1)
    # print('dang qian loss',loss)
    return loss

# @jit(nopython=True)
# def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down,
#                     step_size, beta1, beta2, proj):
#
#     for i in range(batch_size):
#         grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.2
#         # ADAM update
#     mt = mt_arr[indice]
#     mt = beta1 * mt + (1 - beta1) * grad
#     mt_arr[indice] = mt
#
#     vt = vt_arr[indice]
#     vt = beta2 * vt + (1 - beta2) * (grad * grad)
#     vt_arr[indice] = vt
#     epoch = adam_epoch[indice]
#     corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))
#     m = real_modifier.reshape(-1)
#     old_val = m[indice]
#     old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-4)
#     print(np.max(old_val),np.min(old_val))
#     # print(old_val)
#     return old_val
#
#     # set it back to [-0.5, +0.5] region
#     # if proj:
#     #     old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
#     # m[indice] = old_val
#     # adam_epoch[indice] = epoch + 1

@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down, step_size,beta1, beta2, proj):
  for i in range(batch_size):
    grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.1
  # ADAM update
  mt = mt_arr[indice]
  mt = beta1 * mt + (1 - beta1) * grad
  mt_arr[indice] = mt
  vt = vt_arr[indice]
  vt = beta2 * vt + (1 - beta2) * (grad * grad)
  vt_arr[indice] = vt
  epoch = adam_epoch[indice]
  corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
  m = real_modifier.reshape(-1)
  old_val = m[indice]
  old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-8)
  # set it back to [-0.5, +0.5] region
  # if proj:
  #   old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
  m[indice] = old_val
  adam_epoch[indice] = epoch + 1

@jit(nopython=True)
def coordinate_ADAM1(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down, step_size,beta1, beta2, proj):
  # for i in range(batch_size):
  #   grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
  # ADAM update
  mt = mt_arr[indice]
  mt = beta1 * mt + (1 - beta1) * grad

  # print('mt duo da',mt)
  mt_arr[indice] = mt
  vt = vt_arr[indice]
  vt = beta2 * vt + (1 - beta2) * (grad * grad)

  # print('vt duo da',vt)
  vt_arr[indice] = vt
  epoch = adam_epoch[indice]
  corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
  # print('corr shi duo shao',corr)
  m = real_modifier.reshape(-1)
  old_val = m[indice]

  old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-8)
  m[indice] = old_val
  adam_epoch[indice] = epoch + 1


@jit(nopython=True)
def coordinate_Newton(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down,
                      step_size, beta1, beta2, proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
        hess[i] = (losses[i * 2 + 1] - 2 * cur_loss + losses[i * 2 + 2]) / (0.0001 * 0.0001)
    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= step_size * grad / hess
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    m[indice] = old_val


def loss_run(input, target, model, modifier, use_tanh, use_log, targeted, confidence, const):
    if use_tanh:
        # print(modifier[0].shape)
        # print(torch.sum(modifier[0]))
        pert_out = torch.tanh(input + modifier) / 2
    else:
        pert_out = torch.clip((input + modifier) ,0,1)
        # pert_out = (input + modifier)

    output = model(pert_out.permute(0, 3, 1, 2).float())

    if use_tanh:
        loss1 = torch.sum(torch.square(pert_out - torch.tanh(input) / 2), dim=(1, 2, 3))
    else:
        # loss1 = torch.sum(torch.square(pert_out - input), dim=(1, 2, 3))
        loss1 = torch.norm((pert_out - input), p=2, dim=(1, 2, 3))
    loss2 = get_loss(output,target) *const
    # print(loss2)
    # print(loss1)
    l2 = loss1.detach().cpu().numpy()

    loss = loss1.detach().cpu().numpy() + loss2

    return loss, l2, loss2, output.detach().cpu().numpy(), pert_out.detach().cpu().numpy()


def l2_attack(input, target, model, targeted, use_log, use_tanh, solver, reset_adam_after_found=True, abort_early=True,
              batch_size=100, max_iter=500, const=1, confidence=0.0, early_stop_iters=1000, binary_search_steps=10,
              step_size=1, adam_beta1=0.9, adam_beta2=0.999):
    early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iter // 10
    print(input.shape)
    ori_sample = input.transpose(0, 3, 1, 2)
    input = torch.from_numpy(input).cuda()
    ori_sample = torch.from_numpy(ori_sample).float().cuda()

    target = torch.from_numpy(target).cuda()

    ori_label = ((model(ori_sample)>0.5)+0).squeeze()

    if (ori_label==target).all():
        print('Already!!!')
        return input.cpu().numpy()[0],0


    var_len = input.view(-1).size()[0]
    # var_len = 1
    modifier_up = np.zeros(var_len, dtype=np.float32)
    modifier_down = np.zeros(var_len, dtype=np.float32)
    real_modifier = torch.zeros(input.size(), dtype=torch.float32).cuda()


    real = np.zeros(input.size(), dtype=np.float)

    print(real.shape)


    mt = np.zeros(var_len, dtype=np.float32)
    vt = np.zeros(var_len, dtype=np.float32)
    adam_epoch = np.ones(var_len, dtype=np.int32)
    grad = np.zeros(batch_size, dtype=np.float32)
    hess = np.zeros(batch_size, dtype=np.float32)

    mtmy=np.zeros(var_len, dtype=np.float32)
    vtmy=np.zeros(var_len, dtype=np.float32)


    upper_bound = 1e10
    lower_bound = 0.0
    out_best_attack = input.clone()


    out_best_const = const
    out_bestl2 = 1e10
    out_bestscore = -1

    if use_tanh:
        input = torch.atanh(input * 1.99999)

    if not use_tanh:
        modifier_up = 1 - input.clone().detach().view(-1).cpu().numpy()
        modifier_down = 0 - input.clone().detach().view(-1).cpu().numpy()

    def compare(x, y):
        if x.shape==y.shape:
            x = (x > 0.5) + 0
            # print('x',x)
            y = (y > 0.5) + 0
            # print('y',y)
            if (x == y).all():
                return True
            else:
                return False
        else:
            print('x , y not match')

    # with torch.no_grad():
    #     out_sample = input.clone()
    #
    #     qnumm = 0
    #
    #     for step in range(binary_search_steps):
    #         bestl2 = 1e10
    #         prev = 1e6
    #         bestscore =ori_label.cpu().numpy()
    #         last_loss2 = 10.
    #         # reset ADAM status
    #         mtmy.fill(0)
    #         vtmy.fill(0)
    #         adam_epoch.fill(1)
    #
    #
    #         for iter in range(max_iter):
    #
    #
    #             qnumm+=2
    #             var_list = np.array(range(0, var_len), dtype=np.int32)
    #             indice = var_list[np.random.choice(var_list.size, batch_size, replace=False)]
    #
    #             vleft = real.copy()[0].reshape(-1)
    #             vleft[indice]-=0.5
    #             vright = real.copy()[0].reshape(-1)
    #             vright[indice] += 0.5
    #
    #             vl1 = torch.from_numpy(vleft)
    #             vr1 = torch.from_numpy(vright)
    #             vl1 = vl1.view((-1,) + input.size()[1:]).cuda()
    #             vr1 = vr1.view((-1,) + input.size()[1:]).cuda()
    #             con_v = torch.cat((torch.zeros_like(input),vl1, vr1), dim=0)
    #
    #
    #             lossesv, l2sv, losses2v, scoresv, pert_imagesv = loss_run(out_sample, target, model, con_v, use_tanh, use_log,
    #                                                                  targeted,
    #                                                                  confidence, const)
    #             if qnumm % 1 == 0:
    #                 print(qnumm,losses2v[0])
    #             if (iter + 1) % 10 == 0:
    #                 print(losses2v[0])
    #             if losses2v[0]==0:
    #                return  pert_imagesv[0], out_bestscore
    #
    #             grad_of_batch = (lossesv[2] - lossesv[1]) / 1
    #
    #             final_r = real_modifier.clone().detach().cpu().numpy()
    #
    #             coordinate_ADAM1(lossesv, indice, grad_of_batch, hess, batch_size, mtmy, vtmy, final_r, adam_epoch,
    #                             modifier_up, modifier_down, step_size, adam_beta1, adam_beta2, proj=not use_tanh)
    #
    #             final_r = torch.from_numpy(final_r).cuda()
    #
    #             out_sample += final_r
    #
    #             if losses2v[1]==0 or losses2v[2]==0:
    #                 return out_sample[0].cpu().numpy(), out_bestscore
    #
    #
    # return out_sample[0].cpu().numpy(), out_bestscore

####################  YUAN SHI   ######################
    with torch.no_grad():

        qnum = 0
        for step in range(binary_search_steps):
            bestl2 = 1e10
            prev = 1e6
            bestscore =ori_label.cpu().numpy()
            last_loss2 = 100.0
            # reset ADAM status
            mt.fill(0)
            vt.fill(0)
            adam_epoch.fill(1)

            stage = 0

            for iter in range(max_iter):
                if (iter + 1) % 10 == 0:
                    loss, l2, loss2, _, __ = loss_run(input, target, model, real_modifier, use_tanh, use_log, targeted,
                                                      confidence, const)
                    print(qnum)

                    print("[STATS][L2] iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(iter + 1, loss[0],
                                                                                                        l2[0], loss2[0]))
                    sys.stdout.flush()

                var_list = np.array(range(0, var_len), dtype=np.int32)
                indice = var_list[np.random.choice(var_list.size, batch_size, replace=False)]
                var = np.repeat(real_modifier.detach().cpu().numpy(), batch_size * 2 + 1, axis=0)
                for i in range(batch_size):
                    var[i * 2 + 1].reshape(-1)[indice[i]] += 0.5
                    #defult = 0.0001
                    var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.5

                var = torch.from_numpy(var)
                var = var.view((-1,) + input.size()[1:]).cuda()

                losses, l2s, losses2, scores, pert_images = loss_run(input, target, model, var, use_tanh, use_log, targeted,
                                                                     confidence, const)

                qnum += batch_size*2
                if qnum >20000:
                    return pert_images[0], out_bestscore

                if losses2[0] ==0:
                    print('query ',qnum)
                    return pert_images[0], out_bestscore


                real_modifier_numpy = real_modifier.clone().detach().cpu().numpy()
                if solver == "adam":
                    coordinate_ADAM(losses2, indice, grad, hess, batch_size, mt, vt, real_modifier_numpy, adam_epoch,
                                    modifier_up, modifier_down, step_size, adam_beta1, adam_beta2, proj=not use_tanh)

                    # real_modifier_numpy.reshape(-1)[indice] = move *5


                    real_modifier_numpy.reshape(real_modifier.shape)
                    real_modifier = torch.from_numpy(real_modifier_numpy).cuda()

                    out_best_attack+=torch.from_numpy(real_modifier_numpy).cuda()

                    out_best_attack = torch.clip(out_best_attack,0,1)

                if losses2[0] < last_loss2:
                    last_loss2 = losses2[0]




                if losses2[0] == 0.0 and last_loss2 != 0.0 and stage == 0:
                    if reset_adam_after_found:
                        mt.fill(0)
                        vt.fill(0)
                        adam_epoch.fill(1)
                    stage = 1
                if abort_early and (iter + 1) % early_stop_iters == 0:
                    if losses[0] > prev * .9999:
                        print("Early stopping because there is no improvement")
                        break
                    prev = losses[0]

                if l2s[0] < bestl2:
                    bestl2 = l2s[0]

            #
            #     out_bestscore = scores[0]
            #
            # if losses2[0]>last_loss2 and (bestscore== ori_label.cpu().numpy()).all():
            #     print('old constant: ', const)
            #     upper_bound = min(upper_bound, const)
            #     if upper_bound < 1e9:
            #         const = (lower_bound + upper_bound) / 2
            #     print('new constant: ', const)
            # else:
            #     print('old constant: ', const)
            #     lower_bound = max(lower_bound, const)
            #     if upper_bound < 1e9:
            #         const = (lower_bound + upper_bound) / 2
            #     else:
            #         const *= 10
            #     print('new constant: ', const)

    return out_best_attack.cpu().numpy()[0], out_bestscore


def generate_data(test_loader, targeted, samples, start):
    inputs = []
    targets = []
    num_label = 10
    cnt = 0
    for i, data in enumerate(test_loader):
        if cnt < samples:
            if i > start:
                data, label = data[0], data[1]
                if targeted:
                    seq = range(num_label)
                    for j in seq:
                        if j == label.item():
                            continue
                        inputs.append(data[0].numpy())
                        targets.append(np.eye(num_label)[j])
                else:
                    inputs.append(data[0].numpy())
                    targets.append(np.eye(num_label)[label.item()])
                cnt += 1
            else:
                continue
        else:
            break

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def attack(inputs, targets, model, targeted, use_log, use_tanh, solver, device):

    attack, score = l2_attack(np.expand_dims(inputs[0], 0), np.expand_dims(targets[0], 0), model, targeted, use_log,
                                  use_tanh, solver, device)
    return np.array(attack)


if __name__ == '__main__':
    torch.manual_seed(998)
    torch.cuda.manual_seed(998)
    np.random.seed(998)
    use_cuda = True
    use_log = False
    use_tanh = False
    targeted = True
    solver = "adam"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = ML_GCN2007()
    model = get_model('liw2007').to(device)

    npy_path = './npy/liw2007/hidesingle/'

    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    adv_save_path = './adv_save/voc2007/mlliw/hidesingle/'
    if not os.path.exists(adv_save_path):
        os.makedirs(adv_save_path)

    ori_img_path = './ori_test_2007/'

    ori_448_path = './ori_save/voc2007/'
    if not os.path.exists(ori_448_path):
        os.makedirs(ori_448_path)

    ############# ATTACK TYPE ###############################
    gen_adv_file(model, 'hide_single', ori_img_path, npy_path)
    ###   'hide_single'  'hide_all'     'random_case'   'extreme_case'   'person_reduction'  'sheep_augmentation'

    y_t = np.load(os.path.join(npy_path, 'y_target.npy'))

    adv_file_list = os.listdir(ori_img_path)
    adv_file_list.sort(key=lambda x: int(x[-7:-4]))

    with torch.no_grad():
        ii = 1
        for f in adv_file_list:
            if ii > 20:
                break

            num = 30000

            y_target = y_t[ii - 1]

            y_target = np.expand_dims(y_target, axis=0)
            print('target is ',y_target)

            test = Image.open(os.path.join(ori_img_path, f)).resize((448, 448)).convert('RGB')
            test.save(os.path.join(ori_448_path, '{m}.png'.format(m=ii)))
            print('This is num {} pic\n'.format(ii))

            inputs = np.array(test).reshape((1, 448, 448, 3)) / 255.


            timestart = time.time()
            adv = attack(inputs, y_target, model, targeted, use_log, use_tanh, solver, device)
            print('before attack laebl:',
                  (model(torch.tensor((np.array(test).reshape(( 448, 448, 3)) / 255.).transpose(2, 0, 1)).float().unsqueeze(0).cuda()) > 0.5) + 0)
            print('after attack laebl:',(model(torch.tensor(adv.transpose(2,0,1)).float().unsqueeze(0).cuda())>0.5)+0)

            timeend = time.time()
            print("Took", (timeend - timestart) / 60.0, "mins to run", len(inputs), "samples.")


            if not os.path.exists(os.path.join(adv_save_path, '{n}/'.format(n=num))):
                os.makedirs(os.path.join(adv_save_path, '{n}/'.format(n=num)))

            imageio.imwrite(os.path.join(adv_save_path, '{n}/{m}_adv.png'.format(n=num, m=ii)),
                            np.uint8((adv.reshape(448, 448, 3) ) * 255))
            before = np.array(test).reshape(( 448, 448, 3)) / 255.
            after = adv.reshape(448, 448, 3)

            dis = np.linalg.norm(after-before)
            print('adv de norm2 :',dis)
            ii += 1
