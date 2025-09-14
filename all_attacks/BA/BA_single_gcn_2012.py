import os

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import torch

print(torch.cuda.is_available())
print(torch.__version__)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from demo_voc2007_gcn import ML_GCN
import numpy as np
import time
from PIL import Image
import imageio
import torch
from torchvision import transforms
import argparse
import time
import numpy as np

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

# from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image

# from keras.applications.resnet50 import preprocess_input, decode_predictions




def orthogonal_perturbation(delta, prev_sample, target_sample):
    """Generate orthogonal perturbation."""

    perturb = np.random.randn(1, 448, 448, 3)

    perturb /= np.linalg.norm(perturb, axis=(1, 2))
    # print('zui zhi',np.max(perturb),np.min(perturb))
    perturb *= delta * np.mean(get_diff(target_sample, prev_sample))

    # Project perturbation onto sphere around target
    diff = (target_sample.transpose(0, 2, 3, 1) - prev_sample.transpose(0, 2, 3, 1)).astype(np.float32)

    # Orthorgonal vector to sphere surface
    diff /= get_diff(target_sample, prev_sample)  # Orthogonal unit vector

    # We project onto the orthogonal then subtract from perturb
    # to get projection onto sphere surface
    perturb = perturb.transpose(0, 3, 1, 2)
    diff = diff.transpose(0, 3, 1, 2)

    perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff) ** 2) * diff

    # Check overflow and underflow
    perturb = np.clip((perturb + target_sample), 0, 1) - target_sample

    return perturb


def forward_perturbation(epsilon, adv_sample, ori_sample):
    """Generate forward perturbation."""
    perturb = (ori_sample - adv_sample).astype(np.float32)
    perturb *= epsilon
    return perturb


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


def save_image(sample, id, folder,nq):
    """Export image file."""

    sample = np.uint8(sample[0].transpose(1, 2, 0) * 255.)

    sample = Image.fromarray(sample)

    # Save with predicted label for image (may not be adversarial due to uint8 conversion)

    if nq<300:
        if not os.path.exists(os.path.join(folder, '{n}/'.format(n=300))):
            os.makedirs(os.path.join(folder, '{n}/'.format(n=300)))
        imageio.imwrite(os.path.join(folder, '{n}/{m}_adv.png'.format(n=300, m=id)),
                        sample)
        # sample.save(folder, '{n}/{m}_adv.png'.format(n=300, m=id))
        if not os.path.exists(os.path.join(folder, '{n}/'.format(n=3000))):
            os.makedirs(os.path.join(folder, '{n}/'.format(n=3000)))
        imageio.imwrite(os.path.join(folder, '{n}/{m}_adv.png'.format(n=3000, m=id)),
                        sample)
        # sample.save(folder, '{n}/{m}_adv.png'.format(n=3000, m=id))
        if not os.path.exists(os.path.join(folder, '{n}/'.format(n=30000))):
            os.makedirs(os.path.join(folder, '{n}/'.format(n=30000)))
        imageio.imwrite(os.path.join(folder, '{n}/{m}_adv.png'.format(n=30000, m=id)),
                        sample)
        # sample.save(folder, '{n}/{m}_adv.png'.format(n=30000, m=id))
    elif nq < 3000:
        if not os.path.exists(os.path.join(folder, '{n}/'.format(n=3000))):
            os.makedirs(os.path.join(folder, '{n}/'.format(n=3000)))
        imageio.imwrite(os.path.join(folder, '{n}/{m}_adv.png'.format(n=3000, m=id)),
                        sample)
        # sample.save(folder, '{n}/{m}_adv.png'.format(n=3000, m=id))
        if not os.path.exists(os.path.join(folder, '{n}/'.format(n=30000))):
            os.makedirs(os.path.join(folder, '{n}/'.format(n=30000)))
        imageio.imwrite(os.path.join(folder, '{n}/{m}_adv.png'.format(n=30000, m=id)),
                        sample)
        # sample.save(folder, '{n}/{m}_adv.png'.format(n=30000, m=id))
    elif nq < 30000:
        if not os.path.exists(os.path.join(folder, '{n}/'.format(n=30000))):
            os.makedirs(os.path.join(folder, '{n}/'.format(n=30000)))
        imageio.imwrite(os.path.join(folder, '{n}/{m}_adv.png'.format(n=30000, m=id)),
                        sample)


def preprocess(sample_path, size):
    """Load and preprocess image file."""
    img = Image.open(sample_path).resize(size)
    x = np.asarray(img).transpose(2, 0, 1)
    if np.max(x) > 1:
        x = x / 255.
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    return x


def get_diff(sample_1, sample_2):
    """Channel-wise norm of difference between samples."""
    s1 = sample_1.transpose(0, 2, 3, 1)
    s2 = sample_2.transpose(0, 2, 3, 1)

    return np.linalg.norm(s1 - s2, axis=(1, 2))


def boundary_attack(model, x_test, y_target,num,adv_save_path,ini,pic_id):
    # Load model, images and other parameters

    folder = adv_save_path
    l2_threshold = 5
    max_test = num
    best_tem = 1000.
    update = 0


    classifier = model
    print((y_target == 0).all())

    if (y_target == 0 ).all():
        initial_sample = np.random.rand(1, 3, 448, 448)
    else:
        initial_sample = ini


    target_sample = x_test

    attack_class = y_target
    print(attack_class)

    # class type is ndarray

    target_sample_tensor = torch.tensor(target_sample).float().cuda()

    with torch.no_grad():
        attack_class = classifier(torch.tensor(initial_sample).float().cuda())
        target_class = classifier(target_sample_tensor)

    attack_class = attack_class.cpu().numpy()
    attack_class = np.asarray(attack_class)
    attack_class = attack_class.copy()
    attack_class[attack_class >= (0.5 + 0)] = 1
    attack_class[attack_class < (0.5 + 0)] = 0

    target_class = target_class.cpu().numpy()
    target_class = np.asarray(target_class)
    target_class = target_class.copy()
    target_class[target_class >= (0.5 + 0)] = 1
    target_class[target_class < (0.5 + 0)] = 0

    if (target_class==attack_class).all():
        print('Already ADV quit!')
        save_image(target_sample, pic_id, folder, 1)
        return 1

    adversarial_sample = initial_sample

    n_steps = 0
    n_calls = 0
    epsilon = 1.
    delta = 0.01

    # Move first step to the boundary
    while True:
        trial_sample = adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample)

        with torch.no_grad():
            prediction = classifier(torch.tensor(trial_sample).float().cuda())
        prediction = prediction.cpu().numpy()
        prediction = np.asarray(prediction)
        prediction = prediction.copy()
        prediction[prediction >= (0.5 + 0)] = 1
        prediction[prediction < (0.5 + 0)] = 0

        n_calls += 1
        # print(prediction[0].cpu().numpy() == attack_class,prediction,attack_class)

        # print(prediction == attack_class)
        # print(np.all(prediction == attack_class))
        if np.all(prediction == attack_class):
            adversarial_sample = trial_sample
            break
        else:
            epsilon *= 0.9

    # Iteratively run attack
    while True:
        # print("Step #{}...".format(n_steps))
        # # Orthogonal step
        # print("\tDelta step...")

        # Orthogonal step# Orthogonal step# Orthogonal step
        d_step = 0
        while True:
            d_step += 1


            trial_samples = []
            for i in np.arange(10):
                # print(adversarial_sample.shape,)
                trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
                trial_samples.append(trial_sample.squeeze())

            predictions = classifier(torch.tensor(trial_samples).float().cuda())

            predictions = (predictions > 0.5) + 0
            # print(predictions)

            d_score = np.mean(np.all(predictions.cpu().numpy() == attack_class, axis=1))

            n_calls += 10
            if d_score > 0.0:
                if d_score < 0.3:
                    delta *= 0.9
                elif d_score > 0.7:
                    delta /= 0.9
                # print(np.array(trial_samples)[np.all(predictions.cpu().numpy() == attack_class, axis=1)].shape)
                adversarial_sample = np.array(trial_samples)[np.all(predictions.cpu().numpy() == attack_class, axis=1)][0][None]
                # print(adversarial_sample.shape)
                break
            else:
                delta *= 0.9

        # Forward step
        e_step = 0
        while True:

            e_step += 1

            trial_sample = np.clip(adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample),0,1)

            prediction = classifier(torch.tensor(trial_sample).float().cuda())
            prediction = (prediction > 0.5) + 0
            n_calls += 1

            if (prediction[0].cpu().numpy() == attack_class).all():
                adversarial_sample = trial_sample
                epsilon /= 0.5
                break
            elif e_step > 500:
                break
            else:
                epsilon *= 0.5

        n_steps += 1



        with torch.no_grad():
            label = classifier(torch.tensor(adversarial_sample).float().cuda())
        label = (label.cpu().numpy() > 0.5) + 0

        match = np.all(label == attack_class) + 0
        if (match == 1):
            diff = np.mean(get_diff(adversarial_sample, target_sample))
        else:
            save_image(adversarial_sample, pic_id, folder, n_calls)
            break

        if diff < (best_tem-0.2):
            best_tem = diff
            # print('now norm l2', diff)
            save_image(adversarial_sample, pic_id, folder,n_calls)
            print("qur: {}".format(n_calls))
            print("l2: {}".format(diff))
            print("is adv: {}".format((attack_class == prediction[0].cpu().numpy()).all()))
            update = 1

        if update==1 and diff > 2*best_tem:
            break

        if diff < l2_threshold:
            break

        # best_tem = diff

        # if n_calls

        if n_calls >= max_test:
            break


    return n_calls

def gen_adv_file(model, target_type, ori_img_path, npy_path):
    print("generiting target file…")

    advlist = os.listdir(ori_img_path)

    # advlist.sort(key=lambda x: int(x[-7:-4]))
    # coco
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

    # ori_file_list.sort(key=lambda x: int(x[-7:-4]))
    # coco
    ori_file_list.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # ori_file_list.sort(key=lambda x: int(x[0:6]))

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
    adv_pred_match_target = (np.logical_and(
        np.logical_and(np.all((adv_pred == y_target), axis=1),
                       np.asarray(norm) < (77.6 * 1)), np.asarray(norm) > (0.00))) + 0
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
def testresult(model, ori_folder, output_folder, y_target, adv_num):
    print("Load Data from ori_folder")
    X = []
    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
    ])
    ori_imge = os.listdir(ori_folder)

    # ori_imge.sort(key=lambda x: int(x[-7:-4]))
    # coco
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
    torch.manual_seed(998)
    torch.cuda.manual_seed_all(998)
    np.random.seed(998)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###############   MODEL    ###################

    m_ty = "mlgcn"
    modelname = 'gcn2012'
    attackt = "hide_single"
    datasetn = "voc2012"
    inipath = './hidesingle/GCN_2012_adv/'

    qnummmm = 30000

    model = get_model(modelname).to(device)

    npy_path = './npy/{}/{}/'.format(modelname, attackt)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    adv_save_path = './adv_save/{}/{}/{}/'.format(datasetn, m_ty, attackt)
    if not os.path.exists(adv_save_path):
        os.makedirs(adv_save_path)

    # ori_img_path = './{}ori/'.format(datasetn)
    ori_img_path = './hidesingle/ori_2012gcn/'

    ori_448_path = './{}ori448/'.format(datasetn)
    if not os.path.exists(ori_448_path):
        os.makedirs(ori_448_path)

    adv_file_list = os.listdir(ori_img_path)

    adv_file_list.sort(key=lambda x: int(x[0:-8]))
    # coco
    # adv_file_list.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # adv_file_list.sort(key=lambda x: int(x[0:6]))

    ini_list = os.listdir(inipath)

    adv_file_list.sort(key=lambda x: int(x[0:-8]))
    # coco
    # adv_file_list.sort(key=lambda x: int(x[-10:-4]))
    # nuswide
    # adv_file_list.sort(key=lambda x: int(x[0:6]))

    ############# ATTACK TYPE ###############################
    # gen_adv_file(model, 'hide_all', ori_img_path, npy_path)
    ###   'hide_single'  'hide_all'     'random_case'   'extreme_case'   'person_reduction'  'sheep_augmentation'

    ###############   DATA   ###################

    # y_t = np.load(os.path.join(npy_path, 'y_target.npy'))

    # testresult(model, ori_img_path, os.path.join(adv_save_path, '300/'), y_t, 50)

    # #
    for num in [5000]:
        print('NOW  ITER = 300')
        print(num)
        ii = 1
        for f in adv_file_list:
        # for f in adv_file_list:
            if ii > 20:
                break

            if ii >18:
                # print(y_target)
                test = Image.open(os.path.join(ori_img_path, f)).resize((448, 448)).convert('RGB')
                # test.save(os.path.join(ori_448_path, '{m}.png'.format(m=ii)))
                print('This is num {} pic\n'.format(ii))

                # print(os.path.join(adv_folder_path,f))

                x_test = (np.array(test).transpose(2, 0, 1).astype('float32') / 255.).reshape(1, 3, 448, 448)


                ini_img =Image.open(os.path.join(inipath, f)).resize((448, 448)).convert('RGB')
                ini_test = (np.array(ini_img).transpose(2, 0, 1).astype('float32') / 255.).reshape(1, 3, 448, 448)

                # yooo = model(torch.tensor(x_test).float().cuda()).cpu().detach().numpy()
                # yooo =  (yooo>0.5)+0
                # print('y ori is:',yooo)

                # y_target = y_t[ii - 1]
                y_target = model(torch.tensor(ini_test).float().cuda()).cpu().detach().numpy()

                y_target = (y_target > 0.5) + 0

                boundary_attack(model, x_test, y_target,num,adv_save_path,ini_test,ii)


                # img = Image.fromarray(np.uint8(x_adv.squeeze() * 255).transpose(1, 2, 0), 'RGB')
                #
                # if not os.path.exists(os.path.join(adv_save_path, '{n}/'.format(n=num))):
                #     os.makedirs(os.path.join(adv_save_path, '{n}/'.format(n=num)))
                #
                # img.save(os.path.join(adv_save_path, '{n}/{m}_adv.png'.format(n=num, m=ii)))


            ii += 1

