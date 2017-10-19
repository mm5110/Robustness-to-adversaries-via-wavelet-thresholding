from scipy import misc
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np


#Read in image as a numpy array
panda_clean = misc.imread("panda.png")

#EGather basic facts on image
print("Image size is: " + str(panda_clean.shape) + ". Data type of is: " + str(panda_clean.dtype))

#Apply some process to the image to create adversarial image
#Remove a colour - in this case red
panda_nored = panda_clean
panda_nored[:,:,0] = 0

panda_img = plt.imshow(panda_nored)
# plt.show()


#Make green pixels more green
col_mask_loc = panda_clean[:,:,2]>100
panda_moregreen = panda_clean
panda_moregreen[col_mask_loc]=[0,255,0]
panda_img = plt.imshow(panda_moregreen)
plt.show()

#Create a mask:
# print(int(panda_clean.shape[0]))
# print(int(panda_clean.shape[1]))
# mask = np.zeros((int(panda_clean.shape[0]), int(panda_clean.shape[1])))

# panda_adv = panda_clean
# panda_adv[0] = panda_adv[0][1]*mask

#View image using Matplotlib
#panda_img = plt.imshow(panda_arr)
#plt.show()