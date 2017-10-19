#Import libraries
import pywt
from scipy import misc
from scipy import fftpack
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np
import copy


#Define Functions
def calc_2d_wav_image(img, wavelet='db1'):
	print ("Calculating wavelet coefficients of image...")
	coefficients = pywt.wavedec2(img, wavelet)
	print ("Coefficients calculated")
	return coefficients

def SWA(coefficients):
	temp = copy.copy(coefficients)
	return temp

def get_image_from_array(array):
    # img = raw.clip(0, 255)
    img = array.astype('uint8')
    img = im.fromarray(img)
    return img

def calc_2d_iwav_image(coefficients, wavelet='db1'):
	print ("Calculating image from wavelet components...")
	image_array = pywt.waverec2(coefficients, wavelet)
	image = get_image_from_array(image_array)
	print ("Image reconstructed")
	return image, image_array




#Define global parameters
wavelet = 'db1'


#Read in image
panda_clean = im.open("panda.png")


#Convert image to np array
panda_clean_array = np.asarray(panda_clean)


#Calculate the wavelet coefficients of the image
panda_coeffs = pywt.wavedec2(panda_clean_array, wavelet)


#Apply some adversarial function to manipulate the image
panda_coeffs_adv = SWA(panda_coeffs)


#Reconstuct image from perturbed wavelet coefficients
panda_recon, panda_recon_array = calc_2d_iwav_image(panda_coeffs_adv, wavelet)


#Plot to compare output
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(panda_clean)
plt.subplot(1,2,2)
plt.imshow(panda_recon)
plt.show()


