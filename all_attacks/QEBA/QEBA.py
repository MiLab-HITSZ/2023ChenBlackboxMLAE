from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import imageio
from PIL import Image
import math
import os

from PIL import Image
from ml_gcn_model.util import Warp
import torchvision.transforms as transforms


sample = []


def get_lable(arr):
    index = np.argwhere(arr.reshape((-1)) > 0.5).reshape((-1))
    return np.array(index)

def get_lable_(arr):
    index = np.argwhere(arr.reshape((-1)) < 0.5).reshape((-1))
    return np.array(index)




def bi(img,x,y):
    src_h = img.shape[0]
    src_w = img.shape[1]
    dst_h = int(x * src_h)  # 图像缩放倍数
    dst_w = int(y * src_w)  # 图像缩放倍数

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    for c in range(3):
        for h in range(dst_h):
            for w in range(dst_w):
                # 目标点在原图上的位置
                # 使几何中心点重合
                src_x = (w + 0.5) * src_w / dst_w - 0.5
                src_y = (h + 0.5) * src_h / dst_h - 0.5
                if src_x < 0:
                    src_x = 0
                if src_y < 0:
                    src_y = 0
                x1 = int(np.floor(src_x))
                y1 = int(np.floor(src_y))
                x2 = int(min(x1 + 1, src_w - 1))  # 防止超出原图像范围
                y2 = int(min(y1 + 1, src_h - 1.6))

                # x方向线性插值，原公式本来要除一个（x2-x1），这里x2-x1=1
                R1 = (x2 - src_x) * img[y1, x1, c] + (src_x - x1) * img[y1, x2, c]
                R2 = (x2 - src_x) * img[y2, x1, c] + (src_x - x1) * img[y2, x2, c]

                # y方向线性插值，同样，原公式本来要除一个（y2-y1），这里y2-y1=1
                P = (y2 - src_y) * R1 + (src_y - y1) * R2
                dst_img[h, w, c] = P
    return dst_img



def qeba(model,
         sample_,
         sample1,
         clip_max=1,
         clip_min=0,
         constraint='l2',
         num_iterations=10,
         gamma=1.0,
         target_label=None,
         target_image=None,
         stepsize_search='geometric_progression',
         max_num_evals=1e4,
         init_num_evals=100,
         max_q=3000,
         adv_save='',
         verbose=True,
         inum=None):


    global sample

    sample = sample_

    a  = torch.tensor(np.transpose(sample, (2, 0, 1))).unsqueeze(0).cuda().float()

    zero_label = np.asarray(((model(a)>0.5)+0).cpu())*0

    params = {'clip_max': clip_max, 'clip_min': clip_min,
              'shape': sample.shape,
              'shape224':sample1.shape,
              'zero_label': zero_label,
              'target_label': target_label,
              'target_image': target_image,
              'constraint': constraint,
              'num_iterations': num_iterations,
              'gamma': gamma,
              'd': int(np.prod(sample.shape)),
              'stepsize_search': stepsize_search,
              'max_num_evals': max_num_evals,
              'init_num_evals': init_num_evals,
              'max_q': max_q,
              'verbose': verbose,
              'adv_save': adv_save,
              }
    # Set binary search threshold.

    if params['constraint'] == 'l2':
        # params['theta'] = params['gamma'] / (np.sqrt(params['d']) * params['d'])
        params['theta'] = params['gamma'] / np.sqrt(params['d'])
    else:
        params['theta'] = params['gamma'] / (params['d'] ** 2)


    qur_num = 0




    # Initialize.
    perturbed , nq1 = initialize(model, sample, params)


    # Project the initialization to the boundary.
    perturbed_new, dist_post_update , nq2,is_adv = binary_search_batch(sample,
                                                      np.expand_dims(perturbed, 0),
                                                      model,
                                                      params)

    qur_num += (nq1 + nq2)

    if qur_num > params['max_q']:
        print('max qur')
        return perturbed, qur_num



    if not is_adv:
        print("fail get boundary")
        return perturbed, nq2
    else:
        print("update adv")
        perturbed = perturbed_new

    dist = compute_distance(perturbed, sample, constraint)


    shifouxiajiang = 0
    flag = 0

    saveee = 0


    with torch.no_grad():
        for j in np.arange(params['num_iterations']):
            # print("============   start iteration ==================")
            params['cur_iter'] = j + 1

            # Choose delta.
            delta = select_delta(params, dist_post_update)

            # Choose number of evaluations.
            num_evals = int(params['init_num_evals'] * np.sqrt(j + 1))
            num_evals = int(min([num_evals, params['max_num_evals']]))

            # approximate gradient.
            gradf = approximate_gradient(model, perturbed, num_evals,
                                         delta, params)


            qur_num += params["init_num_evals"]

            if params['constraint'] == 'linf':
                update = np.sign(gradf)
            else:
                update = gradf

            # search step size.
            if params['stepsize_search'] == 'geometric_progression':
                # find step size.
                epsilon , nq3 = geometric_progression_for_stepsize(perturbed,
                                                             update, dist, model, params)

                # Update the sample.
                perturbed_new = clip_image(perturbed + epsilon * update,
                                       clip_min, clip_max)

                # Binary search to return to the boundary.
                perturbed_new, dist_post_update,nq4 , is_adv = binary_search_batch(sample,
                                                                  perturbed_new[None], model, params)

                qur_num += (nq3 + nq4)

                if qur_num > 300 and qur_num < 3000 and saveee == 0:
                    if not os.path.exists(os.path.join(params['adv_save'], '{n}/'.format(n=300))):
                        os.makedirs(os.path.join(params['adv_save'], '{n}/'.format(n=300)))

                    imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=300, m=inum)),
                                    np.uint8(perturbed_new * 255.))
                    saveee = 1
                elif qur_num > 3000 and saveee == 1:

                    if not os.path.exists(os.path.join(params['adv_save'], '{n}/'.format(n=3000))):
                        os.makedirs(os.path.join(params['adv_save'], '{n}/'.format(n=3000)))

                    imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=3000, m=inum)),
                                    np.uint8(perturbed_new * 255.))
                    saveee = 0


                if not is_adv:
                    print("fail get boundary")
                    if not os.path.exists(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=300, m=inum))):
                        imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=300, m=inum)),
                                    np.uint8(perturbed_new * 255.))
                    if not os.path.exists(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=3000, m=inum))):
                        imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=3000, m=inum)),
                                        np.uint8(perturbed_new * 255.))

                    if not os.path.exists(os.path.join(params['adv_save'], '{n}/'.format(n=30000))):
                        os.makedirs(os.path.join(params['adv_save'], '{n}/'.format(n=30000)))

                    imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=30000, m=inum)),
                                    np.uint8(perturbed_new * 255.))

                    return perturbed, nq1+nq2+nq3+nq4




                else:
                    print("update adv, use {} quries".format(qur_num))
                    perturbed = perturbed_new
                    if qur_num > params['max_q']:
                        print('max qur')
                        return perturbed, qur_num


            # compute new distance.
            new_dis = compute_distance(perturbed, sample, constraint)
            if new_dis > dist and flag ==0:
                shifouxiajiang+=1
                flag=1
            elif new_dis > dist and flag ==1:
                shifouxiajiang+=1
                print('no better')
            elif new_dis < dist:
                shifouxiajiang = 0
                flag = 0
            if shifouxiajiang>5:
                print('no better for 10 iter so quit!!!')

                if not os.path.exists(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=300, m=inum))):
                    imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=300, m=inum)),
                                    np.uint8(perturbed_new * 255.))
                if not os.path.exists(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=3000, m=inum))):
                    imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=3000, m=inum)),
                                    np.uint8(perturbed_new * 255.))

                if not os.path.exists(os.path.join(params['adv_save'], '{n}/'.format(n=30000))):
                    os.makedirs(os.path.join(params['adv_save'], '{n}/'.format(n=30000)))

                imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=30000, m=inum)),
                                np.uint8(perturbed_new * 255.))
                break

            dist = new_dis

            # dist = compute_distance(perturbed, sample, constraint)

            if verbose:
                print('iteration: {:d}, {:s} distance {:.4}'.format(j + 1, constraint, dist))

            if dist < 5:

                if not os.path.exists(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=300, m=inum))):
                    imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=300, m=inum)),
                                    np.uint8(perturbed_new * 255.))
                if not os.path.exists(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=3000, m=inum))):
                    imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=3000, m=inum)),
                                    np.uint8(perturbed_new * 255.))

                if not os.path.exists(os.path.join(params['adv_save'], '{n}/'.format(n=30000))):
                    os.makedirs(os.path.join(params['adv_save'], '{n}/'.format(n=30000)))

                imageio.imwrite(os.path.join(params['adv_save'], '{n}/{m}_adv.png'.format(n=30000, m=inum)),
                                np.uint8(perturbed_new * 255.))




                return perturbed,nq1+nq2

    num_call = nq1+nq2+nq3+nq4

    return perturbed,num_call


def decision_function(model, images, params):
    """
	Decision function output 1 on the desired side of the boundary,
	0 otherwise.
	注意这里的返回值必须是0，1也就是说return np.all((prob == params['original_label']), axis=1)    + 0     最后要加0
	"""
    images = clip_image(images, params['clip_min'], params['clip_max'])
    imaaa  = torch.tensor(np.transpose(images, (0,3, 1,2,))).cuda().float()
    prob = np.asarray(((model(imaaa) > 0.5) + 0).cpu())
    # print('这个是dec的返回值',np.all((prob == params['original_label']), axis=1) + 0)

    if params['target_label'] is None:
        return np.all((prob == params['zero_label']), axis=1) + 0
    else:
        return np.all((prob == params['target_label']), axis=1) + 0



def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max)


def compute_distance(x_ori, x_pert, constraint='l2'):
    # Compute the distance between two images.
    if constraint == 'l2':
        return np.linalg.norm(x_ori - x_pert)
    elif constraint == 'linf':
        return np.max(abs(x_ori - x_pert))


def approximate_gradient(model, sample, num_evals, delta, params):
    clip_max, clip_min = params['clip_max'], params['clip_min']

    # Generate random vectors.
    noise_shape = [num_evals] + list(params['shape224'])
    # print('zhe lishi noise shape',noise_shape)
    rv_new = np.random.randn(*([num_evals] + list(params['shape'])))

    if params['constraint'] == 'l2':
        rv = np.random.randn(*noise_shape)
        rv_ten =   torch.tensor(np.transpose(rv, (0,3, 1,2,)))
        rv_ten = torch.nn.functional.interpolate(rv_ten, size=[448,448], mode='bilinear', align_corners=True)

        rv = np.transpose(rv_ten.numpy(),(0,2, 3,1,))




    # elif params['constraint'] == 'linf':
    #     rv = np.random.uniform(low=-1, high=1, size=noise_shape)
    #     for i in range(num_evals):
    #         rv_new[i] = bi(rv[i]*255.,2,2)/255.

    rv = rv / np.sqrt(np.sum(rv ** 2, axis=(1, 2, 3), keepdims=True))


    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta

    # query the model.
    decisions = decision_function(model, perturbed, params)
    # print('这里是查询模型的dec',decisions)
    decision_shape = [len(decisions)] + [1] * len(params['shape'])

    fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0


    np.mean(fval)
    # Baseline subtraction (when fval differs)
    if np.mean(fval) == 1.0:  # label changes.
        gradf = np.mean(rv, axis=0)
    elif np.mean(fval) == -1.0:  # label not change.
        gradf = - np.mean(rv, axis=0)
    else:
        fval -= np.mean(fval)
        gradf = np.mean(fval * rv, axis=0)

    # Get the gradient direction.
    gradf = gradf / np.linalg.norm(gradf)

    # print(gradf)
    # print(type(gradf))
    # print(gradf.shape)

    return gradf


def project(original_image, perturbed_images, alphas, params):
    alphas_shape = [len(alphas)] + [1] * len(params['shape'])
    alphas = alphas.reshape(alphas_shape)


    if params['constraint'] == 'l2':
        return (1 - alphas) * original_image + alphas * perturbed_images
    elif params['constraint'] == 'linf':
        out_images = clip_image(
            perturbed_images,
            original_image - alphas,
            original_image + alphas
        )
        return out_images


def binary_search_batch(original_image, perturbed_images, model, params):
    """ Binary search to approach the boundar. """
    # Compute distance between each of perturbed image and original image.
    dists_post_update = np.array([
        compute_distance(
            original_image,
            perturbed_image,
            params['constraint']
        )
        for perturbed_image in perturbed_images])

    # Choose upper thresholds in binary searchs based on constraint.
    if params['constraint'] == 'linf':
        highs = dists_post_update
        # Stopping criteria.
        thresholds = np.minimum(dists_post_update * params['theta'], params['theta'])
    else:
        highs = np.ones(len(perturbed_images))
        thresholds = params['theta']

    lows = np.zeros(len(perturbed_images))

    # Call recursive function.
    nq2 = 0
    num = 0
    while np.max((highs - lows) / thresholds) > 1:
        # if np.max((highs - lows) / thresholds)<5:
        #     lows = 0.5
        #     highs = 1
        #     break
        num = num + 1
        nq2 += 1
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids, params)
        # Update highs and lows based on model decisions.
        decisions = decision_function(model, mid_images, params)

        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)

    out_images = project(original_image, perturbed_images, highs, params)

    # Compute distance of the output image to select the best choice.
    # (only used when stepsize_search is grid_search.)
    dists = np.array([
        compute_distance(
            original_image,
            out_image,
            params['constraint']
        )
        for out_image in out_images])
    idx = np.argmin(dists)

    dist = dists_post_update[idx]
    out_image = out_images[idx]


    out_image = np.clip(out_image,0, 1)

    a_out_image = torch.tensor(np.transpose(out_image, (2, 0, 1))).unsqueeze(0).cuda().float()

    prob = np.asarray(((model(a_out_image) > 0.5) + 0).cpu())


    if params['target_label'] is None:
        if np.all((prob == params['zero_label']), axis=1):
            print("----success bi search----------",np.all((prob == params['zero_label']), axis=1))

            return out_image, dist,nq2,np.all((prob == params['zero_label']), axis=1)
        else:
            print("----fail bi search----------", np.all((prob == params['zero_label']), axis=1))

            return out_image, dist, nq2,np.all((prob == params['zero_label']), axis=1)
    else:
        if np.all((prob == params['target_label']), axis=1):
            print("----success bi search----------",np.all((prob == params['target_label']), axis=1))

            return out_image, dist,nq2,np.all((prob == params['target_label']), axis=1)
        else:
            print("----fail bi search----------", np.all((prob == params['target_label']), axis=1))
            return out_image, dist, nq2,np.all((prob == params['target_label']), axis=1)



def bi_linear(pic,  target_size):
    # 读取输入图像
    th, tw = target_size[0], target_size[1]
    emptyImage = np.zeros(target_size, np.uint8)
    for k in range(3):
        for i in range(th):
            for j in range(tw):
                # 首先找到在原图中对应的点的(X, Y)坐标
                corr_x = (i+0.5)/th*pic.shape[0]-0.5
                corr_y = (j+0.5)/tw*pic.shape[1]-0.5
                # if i*pic.shape[0]%th==0 and j*pic.shape[1]%tw==0:     # 对应的点正好是一个像素点，直接拷贝
                #   emptyImage[i, j, k] = pic[int(corr_x), int(corr_y), k]
                point1 = (math.floor(corr_x), math.floor(corr_y))   # 左上角的点
                point2 = (point1[0], point1[1]+1)
                point3 = (point1[0]+1, point1[1])
                point4 = (point1[0]+1, point1[1]+1)
                fr1 = (point2[1]-corr_y)*pic[point1[0], point1[1], k] + (corr_y-point1[1])*pic[point2[0], point2[1], k]
                fr2 = (point2[1]-corr_y)*pic[point3[0], point3[1], k] + (corr_y-point1[1])*pic[point4[0], point4[1], k]
                emptyImage[i, j, k] = (point3[0]-corr_x)*fr1 + (corr_x-point1[0])*fr2
    return emptyImage



def initialize(model, sample, params):
    """
	Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
	"""
    success = 0
    num_evals = 0

    nq = 0

    if params['target_image'] is None:
        # Find a misclassified random noise.
        while True:
            nq +=1
            random_noise = np.random.uniform(params['clip_min'],
                                             params['clip_max'], size=params['shape'])
            random_noise = (1 - 0.8) * sample + 0.8 * random_noise

            success = decision_function(model, random_noise[None], params)

            num_evals += 1
            if success:
                print('ini successful\n')
                break
            if num_evals>200:
                break
            assert num_evals < 1e4, "Initialization failed! "
            "Use a misclassified image as `target_image`"
        # Binary search to minimize l2 distance to original image.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            nq += 1
            mid = (high + low) / 2.0
            blended = (1 - mid) * sample + mid * random_noise
            success = decision_function(model, blended[None], params)
            if success:
                high = mid
            else:
                low = mid

        initialization = (1 - high) * sample + high * random_noise
    else:
        initialization = params['target_image']
        print("----success ini----------")

    return initialization,nq


def geometric_progression_for_stepsize(x, update, dist, model, params):
    """
	Geometric progression to search for stepsize.
	Keep decreasing stepsize by half until reaching the desired side of the boundary,
	"""
    epsilon = dist / np.sqrt(params['cur_iter'])

    nq3= 0

    def phi(epsilon):
        new = x + epsilon * update
        success = decision_function(model, new[None], params)
        return success

    while not phi(epsilon):
        epsilon /= 2.0
        nq3+=1
        if nq3>20:
            break

    return epsilon,nq3


def select_delta(params, dist_post_update):
    """
	Choose the delta at the scale of distance 
	between x and perturbed sample. 

	"""
    if params['cur_iter'] == 1:
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        if params['constraint'] == 'l2':
            delta = np.sqrt(params['d']) * params['theta'] * dist_post_update
        elif params['constraint'] == 'linf':
            delta = params['d'] * params['theta'] * dist_post_update

    return delta
