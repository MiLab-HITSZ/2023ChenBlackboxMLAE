import numpy as np
import cv2
import math

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
    # 用 CV2 resize函数得到的缩放图像
    # new_img = cv2.resize(pic, (448, 448))
    # cv2.imwrite('QEBA_data/1_cv_img.png', new_img)


import numpy as np
import cv2
from matplotlib import pyplot as plt



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


def main():
    for i in range(20, 36):

        src = './ori_save/voc2007/{n}.png'.format(n=i)

        pic = cv2.imread(src)
        c= bi(pic, 2, 2)


        # pic = cv2.imread(src)
        #
        # c=bi_linear(pic, (224,224,3))
        print(c.shape)
        cv2.imwrite('./QEBA_data/{m}.jpg'.format(m=i), c)

        # pic = cv2.imread(src)
        # data = cv2.resize(pic, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # #放大2倍
        # cv2.imwrite('./QEBA_data/{m}.png'.format(m=i), data)


if __name__ == '__main__':
    main()
