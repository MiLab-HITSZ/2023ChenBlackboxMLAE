import sys
sys.path.append('../')
import numpy as np
import torch
print(torch.cuda.device_count())
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

'''img = Image.open('../IMG/17bicycle_person.jpg')
#img = np.transpose(img, (1, 2, 0))
#img = Image.fromarray(np.uint8(img * 255))
crop = transforms.Resize(299)
img = crop(img)
plt.imshow(img)
plt.show()
print(img)
img = np.asarray(img)
img = img/255
img = img.reshape(1,-1)
img = img + 0.03*np.random.uniform(-1, 1, size=(1,299*447*3)  )
img = img.astype('float32')
#img = np.transpose(img, (2, 0, 1))
img = img.reshape(447,299,3)
img = Image.fromarray(np.uint8(img * 255))
plt.imshow(img)
plt.show()
'''


'''print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
'''

'''
a= np.array([[1,1,1],[2,2,2],[3,3,3]])
b= np.array([[4,4,4],[5,5,5],[6,6,6]])
fitness = np.array([1,0,1,-2,5])
fitness = fitness[:, np.newaxis]
print(np.min(fitness))
if (len(np.where(fitness == 0)[0]) != 0):
    print( a[np.where(fitness == 0)[0][0]])'''

'''i=np.where(a>0.5)
b=np.zeros(a.shape)+1
b[i]=-b[i]

b[np.where(b < 0)] = 0
print(b)
fitness = np.sum(b, axis=1)
# Unconstrained optimization
print(fitness)
fitness = fitness[:, np.newaxis]
print(fitness)
'''
