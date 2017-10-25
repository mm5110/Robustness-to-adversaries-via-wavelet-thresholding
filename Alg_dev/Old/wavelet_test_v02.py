# Intro:
# Purpose of this script is to demonstrate successful packing and unpacking of pywavel;et 


#Import libraries
import pywt
from scipy import misc
from scipy import fftpack
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np
import copy


#Define Functions
def deconstruct_image_array(img_array, wavelet='db1'):
	img_coeffs = []
	for i in range(3):
		temp = calc_2d_wav(img_array[i], wavelet)
		img_coeffs.append(temp)
	return img_coeffs


def reconstruct_image_array(img_coefficients, wavelet='db1'):
	temp1 = []
	for i in range(3):
		temp2 = calc_2d_iwav(img_coefficients[i], wavelet)
		temp1.append(temp2)

	image_recon_array = np.stack(temp1, axis=2)
	return image_recon_array
	

def calc_2d_wav(img, wavelet='db1'):
	print ("Calculating wavelet coefficients of image layer...")
	coefficients = pywt.dwt2(img, wavelet)
	print ("Coefficients calculated")
	return coefficients


def calc_2d_iwav(coefficients, wavelet='db1'):
	print ("Calculating image layer from wavelet coefficients...")
	image_array = pywt.idwt2(coefficients, wavelet)
	print ("Image layer reconstructed")
	return image_array


def SWA(coefficients, perturbation='none'):
	temp = copy.copy(coefficients)

	if perturbation == 'none':
		print('Applying perturbation type: ' + perturbation)
		return temp
	else:
		print('Perturbation type not recognised, no perturbation applied - returing original coefficients')
		return temp


def get_array_from_image(image_path):
	img = im.open(image_path)
	temp =  np.asarray(img)
	temp = swapaxes(temp, 1, 2)
	img_array = np.swapaxes(temp, 0, 2)
	return img_array


def get_image_from_array(arr):
	temp = np.swapaxes(arr, 0, 2)
	temp = np.swapaxes(temp, 1, 2)
	img = temp.clip(0, 255)
	img = img.astype('uint8')
	img = im.fromarray(img)
	return img








#Define global parameters
wavelet = 'db1'


#Select SWA perturbation
# perturbation = 'none'
perturbation = 'dumb'


#Read in image
# panda_clean = im.open("butterfly.png")
panda_clean = np.random.rand(3,8,8)


#Convert image to np array
panda_clean_array = np.asarray(panda_clean)
print(panda_clean_array)



#Calculate the wavelet coefficients of the image from the image array
panda_coeffs = deconstruct_image_array(panda_clean_array, wavelet)


#Unpack wavelet coefficients into a list


#Apply some adversarial function to manipulate the image wavelet coefficients
# panda_coeffs_adv = SWA(panda_coeffs)


# Reconstuct image array from perturbed wavelet coefficients
panda_recon_array = reconstruct_image_array(panda_coeffs, wavelet)
print(panda_clean_array)


# Reconstuct image from array
# 
panda_recon = get_image_from_array(panda_recon_array)


# #Plot to compare output
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(panda_clean)
plt.subplot(1,2,2)
plt.imshow(panda_recon)
plt.show()


