import numpy as np
import torch
import torch.nn.functional as F
import utils
from PIL import Image
import os

class SimBA:
    
    def __init__(self, model, dataset, image_size):
        self.model = model
        self.dataset = dataset
        self.image_size = image_size
    
    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, size, size, 3)
        z = torch.zeros(batch_size, self.image_size, self.image_size, 3)
        z[:, :size, :size, :] = x
        return z
        
    def normalize(self, x):
        return utils.apply_normalization(x, self.dataset)



    # 用于获取相应标签的置信度，为实现多标签各种方式的攻击，改为与目标标签作loss函数
    def get_probs(self, x, y):
        output = self.model(x).cpu()
        print(output)
        print(output - y)
        probs = torch.tensor([torch.sum((output - y)**2)], dtype=torch.double)
        return probs
        # return torch.diag(probs).squeeze(0)

    def get_fit(self, x, y_tar):
        output = self.model(x).cpu().detach().numpy()
        p = np.copy(np.asarray(output))
        q = np.zeros(p.shape) + 0.5
        fit = p-q
        # print(fit)
        fit[:, y_tar] = -fit[:, y_tar]
        fit[np.where(fit < 0)] = 0
        fitness = np.sum(fit, axis=1)
        return fitness


    # 获取标签
    def get_preds(self, x):
        output = self.model.get_prob_(x).cpu()
        print('out',output)
        print('labe;',get_label(output))
        return get_label(output)

    def simba_single(self, x, y, num_iters=1000, epsilon=0.3, targeted=False):
        # print(type(x)) torch.Size([1, 3, 448, 448])
        x_ori = x

        label_x_ori = np.asarray(((self.model(x)>0.5)+0).cpu())
        print('ori label is: ',np.argwhere(label_x_ori.squeeze()!= 0).flatten())

        y_tar = np.argwhere(y.squeeze()!= 0).flatten()
        print('final label: ',y_tar)
        loss = self.get_fit(x, y_tar)
        # print('need reduce label finess:',loss)

        n_dims = x.squeeze().reshape(1, -1)

        perm = torch.randperm(n_dims.shape[1])

        last_prob = loss
        print('initial loss:', last_prob)

        youxiao = 0
        old_loss = 10
        flag = 0


        for i in range(num_iters):
            if last_prob == 0:
                print('already adv,save!')
                return x.squeeze()

            else:
                diff = torch.zeros(n_dims.shape[1])
                diff[perm[i]] = epsilon
                # print(torch.sum(diff))
                # print(x.shape)
                # print((diff.view(x.size()).cuda()).shape)
                left_prob = self.get_fit((x - diff.view(x.size()).cuda()).clamp(0, 1), y_tar)
                if  (left_prob < last_prob):
                    x = (x - diff.view(x.size()).cuda()).clamp(0, 1)
                    # print('chenggong zuo',i)
                    youxiao +=1
                    last_prob = left_prob
                else:
                    right_prob = self.get_fit((x + diff.view(x.size()).cuda()).clamp(0, 1), y_tar)
                    if (right_prob < last_prob):
                        x = (x + diff.view(x.size()).cuda()).clamp(0, 1)
                        # print('chenggong you', i)
                        youxiao += 1
                        last_prob = right_prob

            if i % 200 == 0:
                print('{} check'.format(i),last_prob)
                if last_prob < old_loss:
                    old_loss = last_prob.copy()
                    flag = 0
                else:
                    flag += 1
                if flag > 1:
                    print('NO BETTER!, now flag',flag)
                    return x.squeeze()


        print('final loss:',last_prob)
        lm = x - x_ori
        lmm = np.linalg.norm(np.asarray(lm.cpu()))
        print('norm l2:', lmm)
        return x.squeeze()



    # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # (for targeted attack) <labels_batch>
    def simba_batch(self, images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, linf_bound=0.0,
                    order='rand', targeted=False, pixel_attack=False, log_every=1):
        batch_size = 1
        image_size = 448
        assert self.image_size == image_size
        # sample a random ordering for coordinates independently per batch element
        if order == 'rand':
            indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
        elif order == 'diag':
            indices = utils.diagonal_order(image_size, 3)[:max_iters]
        elif order == 'strided':
            indices = utils.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
        else:
            indices = utils.block_order(image_size, 3)[:max_iters]
        if order == 'rand':
            expand_dims = freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters)
        succs = torch.zeros(batch_size, max_iters)
        queries = torch.zeros(batch_size, max_iters)
        l2_norms = torch.zeros(batch_size, max_iters)
        linf_norms = torch.zeros(batch_size, max_iters)
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: utils.block_idct(z, block_size=image_size, linf_bound=linf_bound)
        remaining_indices = torch.arange(0, batch_size).long()
        for k in range(max_iters):
            dim = indices[k]
            expanded = (images_batch[[0]] + trans(self.expand_vector(x[[0]], expand_dims))).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims))
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded)
            # if len(preds_next) == 0:
            #     print(666)
            #     return expanded, probs, succs, queries, l2_norms, linf_norms
            preds = preds_next
            if targeted:
                # remaining = preds.ne(labels_batch)
                remaining = get_decision(preds, labels_batch)
            else:
                print(preds)
                print(labels_batch)
                print(get_decision(preds, labels_batch))

                remaining = preds.eq(labels_batch)
            # check if all images are misclassified and stop early


            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break

            remaining_indices = torch.arange(0, batch_size)[remaining[0]].long()
            if k > 0:
                succs[:, k-1] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            diff[:, dim] = epsilon
            left_vec = x[[0]] - diff
            right_vec = x[[0]] + diff
            # trying negative direction
            adv = (images_batch[[0]] + trans(self.expand_vector(left_vec, expand_dims))).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[[0]])
            queries_k = torch.zeros(batch_size)
            # increase query count for all images
            queries_k[[0]] += 1
            if targeted:
                improved = left_probs.lt(prev_probs[[0]])
            else:
                improved = left_probs.lt(prev_probs[[0]])
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (images_batch[[0]] + trans(self.expand_vector(right_vec, expand_dims))).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[[0]])
            if targeted:
                right_improved = right_probs.lt(torch.max(prev_probs[[0]], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[[0]], left_probs))
            probs_k = torch.tensor(prev_probs.clone(), dtype=torch.double)
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[[0]] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[[0]] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[[0]] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[[0]] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            if (k + 1) % log_every == 0 or k == max_iters - 1:
                print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                        k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
        preds = self.get_preds(expanded)
        if targeted:
            # remaining = preds.ne(labels_batch)
            remaining = get_decision(preds, labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, max_iters-1] = ~remaining
        return expanded, probs, succs, queries, l2_norms, linf_norms


def get_label(arr):
    index = np.argwhere(arr.reshape((-1)) > 0.5).reshape((-1))
    return np.array(index)

def get_decision(pred, y_target):
    y = get_label(y_target).reshape((-1))
    if len(pred) == len(y):
        if (pred == y).all():
            return torch.tensor([[False]])
    return torch.tensor([[True]])