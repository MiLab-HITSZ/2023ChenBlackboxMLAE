import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append('../yolo_master')
sys.path.append('../boundary_attack_master')
import logging
from boundary_attack_master.boundary_attack_resnet import *


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


class MLSur(object):
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
            l2_sum = 0
            queries = 0
            np.random.seed(1)

            for i in range(len(x_list)):
                best_l2 = 999
                try_time = 0
                giveup = 0
                max_test = 3000
                l2_threshold = 50

                save_image(x_list[i], image_index, 0, save_image_folder)

                while try_time < 50:
                    # Generate initial adversarial examples
                    print("-----------ashjdkakhjajkfgkafhjkafgh agjkhfgjhkae")
                    state = isAdv(self.model, x_list[i], y_target[i])
                    if isAder == False:
                        break
                    Adv = np.random.rand(3, 448, 448)
                    save_image(Adv, image_index, 1, save_image_folder)
                    adv_name = "../boundary_attack_master/images/" + save_image_folder + "/" + str(
                        image_index) + "_adv.png"
                    ori_name = "../boundary_attack_master/images/" + save_image_folder + "/" + str(
                        image_index) + "_ori.png"

                    # Optimizing perturbations by boundary-based attacks
                    l2, calls = boundary_attack(adv_name, ori_name, self.model, max_test, l2_threshold, image_index,
                                                save_image_folder, best_l2)

                    queries = queries + calls
                    print("查询次数：", queries, "=======================================")
                    if queries > 3000:
                        break
                    if l2 < best_l2:
                        best_l2 = l2
                    else:
                        logging.info("The number of failed attempts:" + str(try_time) +
                                     ". The l2 norm of the adversarial samples generated this time is: " + str(l2) +
                                     ". The l2 norm of the current best adversarial example is: " + str(best_l2))
                        try_time += 1
                count += 1
                image_index += 1
                l2_sum += best_l2
            return x_adv, queries


def isAdv( model, image, target):
    x_adv = np.transpose(image, (1, 2, 0))
    x_adv = Image.fromarray(np.uint8(x_adv * 255))
    x_adv = np.asarray(x_adv) / 255.
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

        return True
    else:
        return False

