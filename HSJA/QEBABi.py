import numpy as np
import cv2
import math

def bi_linear(src, dst, target_size):
    pic = cv2.imread(src)       # 读取输入图像
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

    cv2.imwrite(dst, emptyImage)
    # 用 CV2 resize函数得到的缩放图像
    new_img = cv2.resize(pic, (448, 448))
    cv2.imwrite('QEBA_data/1_cv_img.png', new_img)

def main():
    for i in range(20, 36):
        src = './QEBA_data/{n}.png'.format(n=i)
        pic = cv2.imread(src)
        data = cv2.resize(pic, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('QEBA_data/adv/{m}.png'.format(m=i), data)


if __name__ == '__main__':
    main()
