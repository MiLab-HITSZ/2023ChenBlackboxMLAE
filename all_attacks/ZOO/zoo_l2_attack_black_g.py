import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from numba import jit
import math
import time
import scipy.misc
import os
import sys
from PIL import Image

from demo_voc2007_gcn import ML_GCN
import torchvision
import torch.nn as nn
from setup_mnist_model import MNIST
from setup_cifar10_model import CIFAR10

"""##L2 Black Box Attack"""

def get_label(arr):
    index = np.argwhere(arr.reshape((-1)) > 0.5).reshape((-1))
    print(np.array(index))
    return np.array(index)

def get_decision(x, y):
    x = get_label(x)
    print('x',x)
    y = get_label(y)
    print('y', y)
    if len(x) == len(y):
        if (x == y).all():
            return True
    return False


@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down,
                    step_size, beta1, beta2, proj):
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
        # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-8)
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
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
        pert_out = torch.tanh(input + modifier) / 2
    else:
        pert_out = input + modifier
#################### 损失函数第二正则项  ##################
    output = model(pert_out)
    hard_out =((output>0.5)+0)-target
    d_neg = torch.where(hard_out>0.5)
    d_pos = torch.where(hard_out<0.5)
    div_num = modifier.shape[0]
    loss_p = []
    loss_n = []
    step_p = int(len(output[d_pos])/div_num)
    step_n = int(len(output[d_neg])/div_num)
    for i in range(div_num):
        loss_p.append(torch.sum(output[d_pos][i*step_p:(i+1)*step_p]))
        loss_n.append(torch.sum(output[d_neg][i*step_n:(i+1)*step_n]))
    loss_n=torch.tensor(loss_n)
    loss_p = torch.tensor(loss_p)
    loss2 = (loss_n-loss_p).cuda()

    if use_tanh:
        loss1 = torch.sum(torch.square(pert_out - torch.tanh(input) / 2), dim=(1, 2, 3))
    else:
        loss1 = torch.sum(torch.square(pert_out - input), dim=(1, 2, 3))

    # real = torch.sum(target * output, -1)
    # other = torch.max((1 - target) * output - (target * 10000), -1)[0]
    #
    # if use_log:
    #     real = torch.log(real + 1e-30)
    #     other = torch.log(other + 1e-30)

    confidence = torch.tensor(confidence).type(torch.float64).cuda()

    # if targeted:
    #     loss2 = torch.max(other - real, confidence)
    # else:
    #     loss2 = torch.max(real - other, confidence)
    # loss2 = const * loss2

    # loss2 = const * torch.sum(torch.square(target - output), dim=(1))

    l2 = loss1
    loss = loss1 + loss2*const

    return loss.detach().cpu().numpy(), l2.detach().cpu().numpy(), loss2.detach().cpu().numpy(), output.detach().cpu().numpy(), pert_out.detach().cpu().numpy()


def l2_attack(input, target, model, targeted, use_log, use_tanh, solver, reset_adam_after_found=True, abort_early=True,
              batch_size=1, max_iter=300, const=1, confidence=0.0, early_stop_iters=50, binary_search_steps=6,
              step_size=0.1, adam_beta1=0.9, adam_beta2=0.999):
    # 1200-3w chaxun.40-1000cha
    early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iter // 10

    input = torch.from_numpy(input).cuda()
    target = torch.from_numpy(target).cuda()



    var_len = input.view(-1).size()[0]
    # var_len = 1
    modifier_up = np.zeros(var_len, dtype=np.float32)
    modifier_down = np.zeros(var_len, dtype=np.float32)
    real_modifier = torch.zeros(input.size(), dtype=torch.float32).cuda()
    mt = np.zeros(var_len, dtype=np.float32)
    vt = np.zeros(var_len, dtype=np.float32)
    adam_epoch = np.ones(var_len, dtype=np.int32)
    grad = np.zeros(batch_size, dtype=np.float32)
    hess = np.zeros(batch_size, dtype=np.float32)

    upper_bound = 1e10
    lower_bound = 0.0
    out_best_attack = input.clone().detach().cpu().numpy()
    out_best_const = const
    out_bestl2 = 1e10
    out_bestscore = -1

    if use_tanh:
        input = torch.atanh(input * 1.99999)

    if not use_tanh:
        modifier_up = 0.5 - input.clone().detach().view(-1).cpu().numpy()
        modifier_down = -0.5 - input.clone().detach().view(-1).cpu().numpy()

    def compare(x, y):
        x = (x>0.5)+0
        y= y.reshape((-1))
        if targeted:
            return (x==y).all()
            # return x == y
        else:
            return (x != y).any()

    for step in range(binary_search_steps):
        bestl2 = 1e10
        prev = 1e6
        bestscore = -1
        last_loss2 = 1.0
        # reset ADAM status
        mt.fill(0)
        vt.fill(0)
        adam_epoch.fill(1)
        stage = 0

        for iter in range(max_iter):
            if (iter + 1) % 100 == 0:
                loss, l2, loss2, _, __ = loss_run(input, target, model, real_modifier, use_tanh, use_log, targeted,
                                                  confidence, const)
                print("[STATS][L2] iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(iter + 1, loss[0],
                                                                                                    l2[0], loss2[0]))
                sys.stdout.flush()

            var_list = np.array(range(0, var_len), dtype=np.int32)
            indice = var_list[np.random.choice(var_list.size, batch_size, replace=False)]
            var = np.repeat(real_modifier.detach().cpu().numpy(), batch_size * 2 + 1, axis=0)
            for i in range(batch_size):
                var[i * 2 + 1].reshape(-1)[indice[i]] += 0.001
                var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.001
            var = torch.from_numpy(var)
            var = var.view((-1,) + input.size()[1:]).cuda()

            # print(var[0])

            losses, l2s, losses2, scores, pert_images = loss_run(input, target, model, var, use_tanh, use_log, targeted,
                                                                 confidence, const)


            real_modifier_numpy = real_modifier.clone().detach().cpu().numpy()
            if solver == "adam":
                coordinate_ADAM(losses, indice, grad, hess, batch_size, mt, vt, real_modifier_numpy, adam_epoch,
                                modifier_up, modifier_down, step_size, adam_beta1, adam_beta2, proj=not use_tanh)
            if solver == "newton":
                coordinate_Newton(losses, indice, grad, hess, batch_size, mt, vt, real_modifier_numpy, adam_epoch,
                                  modifier_up, modifier_down, step_size, adam_beta1, adam_beta2, proj=not use_tanh)
            real_modifier = torch.from_numpy(real_modifier_numpy).cuda()

            if losses2[0] == 0.0 and last_loss2 != 0.0 and stage == 0:
                if reset_adam_after_found:
                    mt.fill(0)
                    vt.fill(0)
                    adam_epoch.fill(1)
                stage = 1
            last_loss2 = losses2[0]

            if abort_early and (iter + 1) % early_stop_iters == 0:
                if losses[0] > prev * .9999:
                    print("Early stopping because there is no improvement")
                    break
                prev = losses[0]

            if l2s[0] < bestl2 and compare(scores[0], target.cpu().numpy()):
                bestl2 = l2s[0]
                bestscore = scores[0]

            if l2s[0] < out_bestl2 and compare(scores[0], target.cpu().numpy()):
                if out_bestl2 == 1e10:
                    print(
                        "[STATS][L3](First valid attack found!) iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(
                            iter + 1, losses[0], l2s[0], losses2[0]))
                    sys.stdout.flush()


                out_bestl2 = l2s[0]
                out_bestscore = scores[0]
                out_best_attack = pert_images[0]
                out_best_const = const
        print(bestscore)
        if compare(bestscore, target.cpu().numpy()) and bestscore != -1:
            print('old constant: ', const)
            upper_bound = min(upper_bound, const)
            if upper_bound < 1e9:
                const = (lower_bound + upper_bound) / 2
            print('new constant: ', const)
        else:
            print('old constant: ', const)
            lower_bound = max(lower_bound, const)
            if upper_bound < 1e9:
                const = (lower_bound + upper_bound) / 2
            else:
                const *= 10
            print('new constant: ', const)

    return out_best_attack, out_bestscore


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
    r = []
    print('go up to', len(inputs))
    # run 1 image at a time, minibatches used for gradient evaluation
    for i in range(len(inputs)):
        print('tick', i + 1)
        attack, score = l2_attack(np.expand_dims(inputs[i], 0), np.expand_dims(targets[i], 0), model, targeted, use_log,
                                  use_tanh, solver, device)
        r.append(attack)
    return np.array(r)


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    # a = torch.tensor([[1,2],[3,4]])
    # b = torch.tensor([[[1,2],[3,4]],[[1,2],[3,4]],[[1,2],[3,4]]])
    # print(a+b)


    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # test_set = datasets.MNIST(root = './data', train=False, transform = transform, download=True)
    # test_set = datasets.CIFAR10(root = './data', train=False, transform = transform, download=True)
    # test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=True)

    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # model = MNIST().to(device)
    # model = CIFAR10().to(device)
    #
    # # model.load_state_dict(torch.load('./models/mnist_model.pt'))
    # model.load_state_dict(torch.load('./models/cifar10_model.pt'))
    # model.eval()
    model = ML_GCN()

    use_log = True
    use_tanh = True
    targeted = True
    solver = "newton"
    # start is a offset to start taking sample from test set
    # samples is the how many samples to take in total : for targeted, 1 means all 9 class target -> 9 total samples whereas for untargeted the original data
    # sample is taken i.e. 1 sample only
    # inputs, targets = generate_data(test_loader,targeted,samples=1,start=6)
    # inputs = sample = np.array(
    #     Image.open('data/voc/VOCdevkit/VOC2007/JPEGImages/000008.jpg').resize((448, 448)).convert('RGB')).reshape(
    #     (1, 448, 448, 3)) / 255. - 0.5

    for i in range(10,99):
        inputs = sample = np.array(
            Image.open('data/voc/VOCdevkit/VOC2007/JPEGImages/0000{n}.jpg'.format(n=i)).resize((448, 448)).convert('RGB')).reshape(
            (1, 448, 448, 3)) / 255. - 0.5
        targets = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        timestart = time.time()
        adv = attack(inputs, targets, model, targeted, use_log, use_tanh, solver, device)
        # print(adv.shape)
        # print(adv[0].shape)
        # print(adv)
        Image.fromarray(np.uint8((adv[0].reshape(448, 448, 3) + 0.5 ) * 255)).save('./advtest/{m}.png'.format(m=i))
        timeend = time.time()
        print("Took", (timeend - timestart) / 60.0, "mins to run", len(inputs), "samples.")

    if use_log:
        valid_class = np.argmax(F.softmax(model(torch.from_numpy(inputs).cuda()), -1).detach().cpu().numpy(), -1)
        adv_class = np.argmax(F.softmax(model(torch.from_numpy(adv).cuda()), -1).detach().cpu().numpy(), -1)

    else:
        valid_class = np.argmax(model(torch.from_numpy(inputs).cuda()).detach().cpu().numpy(), -1)
        adv_class = np.argmax(model(torch.from_numpy(adv).cuda()).detach().cpu().numpy(), -1)

    acc = ((valid_class == adv_class).sum()) / len(inputs)
    print("Valid Classification: ", valid_class)
    print("Adversarial Classification: ", adv_class)
    print("Success Rate: ", (1.0 - acc) * 100.0)
    print("Total distortion: ", np.sum((adv - inputs) ** 2) ** .5)

    # for saving the mnist samples
    # for i in range(len(inputs)):
    #   save(inputs[i], "original_"+str(i)+".png")
    #   save(adv[i], "adversarial_"+str(i)+".png")
    #   save(adv[i] - inputs[i], "diff_"+str(i)+".png")

    # visualization of created mnist adv examples
    # cnt=0
    # plt.figure(figsize=(10,10))
    # for i in range(len(adv)):
    #   cnt+=1
    #   plt.subplot(10,10,cnt)
    #   plt.xticks([], [])
    #   plt.yticks([], [])
    #   plt.title("{} -> {}".format(valid_class[i],adv_class[i]))
    #   plt.imshow(adv[i].reshape(28,28), cmap="gray")
    # plt.tight_layout()
    # if targeted:
    #   if solver=="newton":
    #     plt.savefig('newton_targeted_mnist.png')
    #   else:
    #     plt.savefig('adam_targeted_mnist.png')
    # else:
    #   if solver=="newton":
    #   plt.savefig('newton_untargeted_mnist.png')
    # else:
    #   plt.savefig('adam_untargeted_mnist.png') 

    # visualization of created cifar10 adv examples
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    cnt = 0
    plt.figure(figsize=(10, 10))
    for i in range(len(adv)):
        cnt += 1
        plt.subplot(10, 10, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("{}->{}".format(classes[valid_class[i]], classes[adv_class[i]]))
        plt.imshow(((adv[i] + 0.5)).transpose(1, 2, 0))
    plt.tight_layout()
    if targeted:
        if solver == "newton":
            plt.savefig('newton_targeted_cifar10.png')
        else:
            plt.savefig('adam_targeted_cifar10.png')
    else:
        if solver == "newton":
            plt.savefig('newton_untargeted_cifar10.png')
        else:
            plt.savefig('adam_untargeted_cifar10.png')
