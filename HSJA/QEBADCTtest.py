import cv2
import numpy as np
import matplotlib.pyplot as plt


# 整张图 DCT 变换
def whole_img_dct(img_f32):
    img_dct = cv2.dct(img_f32)            # 进行离散余弦变换
    img_dct_log = np.log(abs(img_dct))    # 进行log处理
    img_idct = cv2.idct(img_dct)          # 进行离散余弦反变换
    return img_dct_log, img_idct

# 分块图 DCT 变换
def block_img_dct(img_f32):
    height,width = img_f32.shape[:2]
    block_y = height // 8
    block_x = width // 8
    height_ = block_y * 8
    width_ = block_x * 8
    img_f32_cut = img_f32[:height_, :width_]
    img_dct = np.zeros((height_, width_), dtype=np.float32)
    new_img = img_dct.copy()
    for h in range(block_y):
        for w in range(block_x):
            # 对图像块进行dct变换
            img_block = img_f32_cut[8*h: 8*(h+1), 8*w: 8*(w+1)]
            img_dct[8*h: 8*(h+1), 8*w: 8*(w+1)] = cv2.dct(img_block)

            # 进行 idct 反变换
            dct_block = img_dct[8*h: 8*(h+1), 8*w: 8*(w+1)]
            img_block = cv2.idct(dct_block)
            new_img[8*h: 8*(h+1), 8*w: 8*(w+1)] = img_block
    img_dct_log2 = np.log(abs(img_dct))
    return img_dct_log2, new_img

 # ouput img properties
def funOutputImgProperties(img):
    print("properties:shape:{},size:{},dtype:{}".format(img.shape,img.size,img.dtype))


if __name__ == '__main__':
    img3Cha = cv2.imread('home/czj/jjr/HSJA-master/data/voc/VOCdevkit/VOC2007/JPEGImages/000001.jpg', cv2.IMREAD_COLOR)
    cv2.imshow('IMREAD_COLOR+Color', img3Cha)
    cv2.waitKey()
    funOutputImgProperties(img3Cha)

    img = cv2.imread('home/czj/jjr/HSJA-master/data/voc/VOCdevkit/VOC2007/JPEGImages/000001.jpg', 0)
    # 数据类型转换 转换为浮点型
    print('0\n', img)
    img1 = img.astype(np.float)

    # 进行离散余弦变换
    img_dct = cv2.dct(img1)
    print('1\n', img_dct)
    # 进行log处理
    img_dct_log = np.log(abs(img_dct))
    print('2\n', img_dct_log)
    # 进行离散余弦反变换
    img_idct = cv2.idct(img_dct)
    print('3\n', img_idct)
    # res = img_idct.astype(np.uint8) # 浮点型转整型 小数部分截断
    # print('3-1\n',res)

    plt.subplot(131)
    plt.imshow(img, 'gray')
    plt.title('original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132)
    plt.imshow(img_dct_log)
    plt.title('DCT'), plt.xticks([]), plt.yticks([])
    plt.subplot(133)
    plt.imshow(img_idct, 'gray')
    plt.title('IDCT'), plt.xticks([]), plt.yticks([])
    plt.show()
