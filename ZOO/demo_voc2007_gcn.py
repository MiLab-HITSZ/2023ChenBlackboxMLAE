import argparse

import numpy as np
import torch.nn as nn

import utils
from engine import *
from models import *
from voc import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ML_GCN(nn.Module):

    def __init__(self):
        super(ML_GCN, self).__init__()
        self.query_num = 0
        num_classes = 20

        # load model
        self.model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/voc/voc_adj.pkl')

        state = {'batch_size': 1, 'image_size': 448, 'max_epochs': 0,
                 'evaluate': 0, 'resume': 'checkpoint/voc/voc_checkpoint.pth.tar', 'num_classes': 20,
                 'difficult_examples': True,
                 'save_model_path': 'checkpoint/voc2007/', 'workers': 0, 'epoch_step': 0,
                 'lr': 0.1}

        self.engine = GCNMultiLabelMAPEngine(state)

        with open('data/voc/voc_glove_word2vec.pkl', 'rb') as f:
            self.inp = torch.tensor(pickle.load(f)).unsqueeze(0).cuda()

        normalize = transforms.Normalize(mean=self.model.image_normalization_mean,
                                         std=self.model.image_normalization_std)
        self.transform = transforms.Compose([
            Warp(448),
            transforms.ToTensor(),
            normalize,
        ])

        self.engine.init_learning(self.model)

        checkpoint = torch.load(state['resume'])
        self.model.load_state_dict(checkpoint['state_dict'])

        cudnn.benchmark = True
        self.model = torch.nn.DataParallel(self.model).cuda()

    def loss(self, y, logits):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        loss = np.sum((y - logits) * (y - logits))

        return loss.flatten()

    def forward(self, x):
        return torch.tensor(self.get_prob_(x.cpu().detach().numpy()))

    def get_prob(self, img):
        img = Image.fromarray(np.uint8(img))
        img = img.convert('RGB')
        img = self.transform(img).unsqueeze(0).cuda()
        input = (img, "img", self.inp)
        self.engine.validate(model=self.model, input_=input)
        return self.engine.state['output'].cpu().detach().numpy()

    def get_prob_(self, img):
        with torch.no_grad():
            if img.shape[0] == 1:
                self.query_num += 1
                img = Image.fromarray(np.uint8((img[0] + 0.5) * 255.))
                img = img.convert('RGB')
                img = self.transform(img).unsqueeze(0).cuda()
                input = (img, "img", self.inp)
                self.engine.validate(model=self.model, input_=input)
                if self.query_num % 500 == 0:
                    print('查询次数:', self.query_num)
                    print('置信度:\n', self.engine.state['output'].cpu().detach().numpy())
                return torch.tensor(self.engine.state['output'].cpu().detach().numpy()).cuda()
            else:
                output = None
                for i in range(img.shape[0]):
                    self.query_num += 1
                    img_ = Image.fromarray(np.uint8((img[i] + 0.5) * 255.))
                    img_ = img_.convert('RGB')
                    img_ = self.transform(img_).unsqueeze(0).cuda()
                    input = (img_, "img", self.inp)
                    self.engine.validate(model=self.model, input_=input)
                    if self.query_num % 500 == 0:
                        print('查询次数:', self.query_num)
                        print('置信度:\n', self.engine.state['output'].cpu().detach().numpy())
                    if i == 0:
                        output = self.engine.state['output'].cpu().detach().numpy().reshape((1, 20))
                    else:
                        output = np.hstack((output, self.engine.state['output'].cpu().detach().numpy()))
                return torch.tensor(output.reshape((-1, 20))).cuda()

    def test(self, img_path):
        with torch.no_grad():
            img = Image.open(img_path).resize((448, 448)).convert('RGB')
            img = self.transform(img).unsqueeze(0).cuda()
            input = (img, "img", self.inp)
            self.engine.validate(model=self.model, input_=input)
            print(self.engine.state['output'])
            return self.engine.state['output'].cpu().detach().numpy()


if __name__ == '__main__':
    mlgcn = ML_GCN()
    mlgcn.test("data/voc/VOCdevkit/VOC2007/JPEGImages/000002.jpg")
    mlgcn.test("2.png")
