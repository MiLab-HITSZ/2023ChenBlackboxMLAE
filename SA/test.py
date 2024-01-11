import numpy as np
from PIL import Image


img = Image.open('data/voc/VOCdevkit/VOC2007/JPEGImages/006668.jpg').resize((448,448)).convert('RGB')
img = np.array(img)
print(img)
img_ = img
print('\n\n')
img = Image.fromarray(np.uint8(img), 'RGB')
# img.save('test.png', icc_profile=img.info.get('icc_profile'), quality=100)
img.save('test.jpg')
img = np.array(Image.open('test.png').resize((448,448)).convert('RGB'))
print((img == img_).all())
print(img)