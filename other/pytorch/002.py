'''
import matplotlib.pyplot as plt

img = plt.imread('image/002.jpg')
print(img.shape)
print(img.dtype)
print(type(img))
plt.imshow(img)
plt.show()
'''

'''
from PIL import Image
img1 = Image.open('image/002.jpg')
print(type(img1))
img1.show()
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img = Image.open('image/002.jpg')
img_ndarray = np.array(img)
plt.imshow(img_ndarray)
plt.show()
