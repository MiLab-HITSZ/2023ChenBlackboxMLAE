import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append('../yolo_master')
sys.path.append('../boundary_attack_master')
import logging
from boundary_attack_master.boundary_attack_resnet import *

def save_json_fall(name,save_image_folder):
    ans_list={
        "name":name,
        "l1_norm":0,
        "l2_norm":0,
        "infiy_norm":0,
        "rmsd":0,
        "mean":0
    }
    if os.path.exists("../boundary_attack_master/JSON/"+save_image_folder+"/"+str(name)+".json"):
        return
    else:
        with open("../boundary_attack_master/JSON/"+save_image_folder+"/"+str(name)+".json", "w") as f:
            json.dump(ans_list, f)

def save_image(image,index,isadv,save_image_folder):
    sample=image*255
    sample = torch.tensor(sample)
    image = sample.permute(1,2,0)
    image = image.cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    if isadv == 1 :
        image.save(os.path.join("../boundary_attack_master","images",save_image_folder, "{}_adv.png".format(index)))
    else:
	    image.save(os.path.join("../boundary_attack_master","images", save_image_folder,"{}_ori.png".format(index)))

class MLBA(object):
    def __init__(self, model):
        self.model = model

    def generate_np(self, x_list, ori_loader, **kwargs):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        logging.info('prepare attack')
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']
        batch_size = kwargs['batch_size']
        yolonet = kwargs['yolonet']
        image_index = kwargs['image_index']
        save_image_folder = kwargs['save_image_folder']
        x_adv = []
        success = 0
        count = 0
        l2_sum=0
        queries = 0
        np.random.seed(1)

        for i in range(len(x_list)):
            best_l2 = 999
            try_time = 0
            giveup = 0
            max_test = 3000
            l2_threshold = 80

            save_image(x_list[i], image_index, 0, save_image_folder)

            # Adv = np.random.rand(3, 448, 448)*0.7
            # np.seterr(divide='ignore', invalid='ignore')
            # Adv = np.clip(Adv, 0.3, 0.7)

            # np.seterr(divide='ignore', invalid='ignore')
            while try_time < 10:
                # Generate initial adversarial examples
                print("-----------ashjdkakhjajkfgkafhjkafgh agjkhfgjhkae")
                Adv, isAder, giveup = GenBig(self.model, x_list[i], y_target[i], ori_loader, try_time, yolonet, giveup)
                if isAder == False:
                    save_json_fall(image_index, save_image_folder)
                    break
                # hide all

                save_image(Adv, image_index, 1, save_image_folder)
                adv_name = "../boundary_attack_master/images/" + save_image_folder + "/" + str(image_index) + "_adv.png"
                ori_name = "../boundary_attack_master/images/" + save_image_folder + "/" + str(image_index) + "_ori.png"

                # Optimizing perturbations by boundary-based attacks
                l2, calls = boundary_attack(adv_name, ori_name, self.model, max_test, l2_threshold, image_index,
                                            save_image_folder, best_l2)

                queries = queries + calls
                print("查询次数：", queries, "=======================================")
                if l2 < l2_threshold:
                    break
                if queries > 3000:
                    break
                # if l2 < best_l2:
                #     best_l2 = l2
                else:
                    logging.info("The number of failed attempts:" + str(try_time) +
                                 ". The l2 norm of the adversarial samples generated this time is: " + str(l2) +
                                 ". The l2 norm of the current best adversarial example is: " + str(best_l2))
                    try_time += 1
            count += 1
            image_index += 1
            l2_sum += best_l2
        return x_adv , queries


def IOU(put,box):
    s_box = (box[2] - box[0]) * (box[3] - box[1])
    s_put = ( put[2]-put[0])*(put[3]-put[1])
    s_overlap = (min(box[3],put[3])-max(box[1],put[1]))*(min(box[2],put[2])-max(box[0],put[0]))
    iou = max(0, s_overlap / (s_box + s_put - s_overlap))
    return iou

def GenBig( model, image, target,ori_loader,jump,yolonet,quit):
    img_dir = "../data/voc2007/VOCdevkit/VOC2007/JPEGImages/"
    imglist = os.listdir(img_dir)
    boxes = []
    target_label = np.argwhere(target > 0)
    time = 0
    giveup = quit
    if quit !=1:
        while time in range(200):
            x_adv = np.transpose(image, (1, 2, 0))
            x_adv = Image.fromarray(np.uint8(x_adv * 255))
            for num in range(len(target_label)):
                flag = target_label[num][0]
                file = open("../data/voclabels.txt")
                for i in range(flag):
                    skip = file.readline()
                lab = file.readline().split('\n')[0]
                seed = np.random.uniform(0,9000)
                count = 0
                for img_name in imglist:
                    if count<seed:
                        count+=1
                        continue
                    img = Image.open('../data/voc2007/VOCdevkit/VOC2007/JPEGImages/' + img_name)
                    windows=yolonet.detect_image(img)
                    find = 0
                    for win in  windows:
                        if win[0] == lab:
                            pos = []
                            pos.append(win[1])
                            pos.append(win[2])
                            pos.append(win[3])
                            pos.append(win[4])
                            find=1
                            break
                    if find==1:
                        part = img.crop(pos)
                        break
                for i in range(20):
                    x1 = int(np.random.uniform(448-part.size[0]))
                    y1 = int(np.random.uniform(448-part.size[1]))
                    put = [x1, y1,x1+part.size[0],y1+part.size[1]]
                    can = True
                    for box in boxes:
                        if (put[0]>box[0] and put[0]<box[2] ) or (put[1]>box[1] and put[1]<box[3]):
                            can = False
                    if can == True:
                        boxes.append(put)
                        break
                x_adv.paste(part, put)
            x_adv = np.asarray(x_adv)/255.
            x_adv = np.transpose(x_adv, (2, 0, 1))
            x_adv = np.clip(np.tile(x_adv, (1, 1, 1, 1)), 0., 1.)
            with torch.no_grad():
                if torch.cuda.is_available():
                    predict = model(torch.tensor(x_adv, dtype=torch.float32).cuda()).cpu()
                else:
                    predict = model(torch.tensor(x_adv, dtype=torch.float32))
            ori_pred = np.asarray(predict)
            pred = ori_pred[0].copy()
            pred[pred >= (0.5 + 0)] = 1
            pred[pred < (0.5 + 0)] = -1
            adv_pred_match_target = np.all((pred == target))
            if adv_pred_match_target:
                x_adv = x_adv.squeeze()
                logging.info(("Successfully generated initial adversarial example"))
                print(x_adv)
                print(type(x_adv))
                print(x_adv.shape)

                return x_adv, True, giveup
            else:
                time+=1

    # Unable to generate initial adversarial samples by YOLO object detector
    # Try directly select initial adversarial examples from the surrogate dataset
    giveup = 1
    logging.info("Unable to generate initial adversarial samples by YOLO object detector")
    count = 0
    for i, (input, label) in enumerate(ori_loader):
        x_list = input[0].cpu().numpy()
        label = np.asarray(label)
        for j in range(label.shape[0]):
            adv_pred_match_target = np.all((label[j] == target))
            if adv_pred_match_target:
                if count < jump:
                    count += 1
                    continue
                logging.info('Successfully found an initial adversarial example in surrogate dataset')
                return x_list[j], True, giveup


    return x_list[0], False,giveup


def black_grad_batch(x_boundary, q_max, sigma, random_noises, batch_size, original_label):
    grad_tmp = []  # estimated gradients in each estimate_batch
    z = []  # sign of grad_tmp
    outs = []
    num_batchs = math.ceil(q_max / batch_size)
    last_batch = q_max - (num_batchs - 1) * batch_size
    EstNoise = SubNoise(batch_size, sub_basis_torch).cuda()
    all_noises = []
    for j in range(num_batchs):
        if j == num_batchs - 1:
            EstNoise_last = SubNoise(last_batch, sub_basis_torch).cuda()
            current_batch = EstNoise_last()
            current_batch_np = current_batch.cpu().numpy()
            noisy_boundary = [x_boundary[0, :, :, :].cpu().numpy()] * last_batch + sigma * current_batch.cpu().numpy()

        else:
            current_batch = EstNoise()
            current_batch_np = current_batch.cpu().numpy()
            noisy_boundary = [x_boundary[0, :, :, :].cpu().numpy()] * batch_size + sigma * current_batch.cpu().numpy()

        all_noises.append(current_batch_np)

        noisy_boundary_tensor = torch.tensor(noisy_boundary).to(device)

        predict_labels = torch.argmax(net.forward(noisy_boundary_tensor), 1).cpu().numpy().astype(int)

        outs.append(predict_labels)
    all_noise = np.concatenate(all_noises, axis=0)
    outs = np.concatenate(outs, axis=0)

    for i, predict_label in enumerate(outs):
        if predict_label == original_label:
            z.append(1)
            grad_tmp.append(all_noise[i])
        else:
            z.append(-1)
            grad_tmp.append(-all_noise[i])

    grad = -(1 / q_max) * sum(grad_tmp)

    grad_f = torch.tensor(grad).to(device)[None, :, :, :]

    return grad_f, sum(z)


def bin_search(x_0, x_random, tol):
    num_calls = 0
    adv = x_random
    cln = x_0

    while True:

        mid = (cln + adv) / 2.0
        num_calls += 1

        if is_adversarial(mid, orig_label):
            adv = mid
        else:
            cln = mid

        if torch.norm(adv - cln).cpu().numpy() < tol:
            break

    return adv, num_calls