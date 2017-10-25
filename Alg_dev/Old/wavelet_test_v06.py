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
	img_coeffs_array = []
	img_coeffs_slices = []
	for i in range(3):
		temp1 = pywt.wavedec2(img_array[i], wavelet)
		temp2, temp3 = pywt.coeffs_to_array(temp1)
		img_coeffs.append(temp1)
		img_coeffs_array.append(temp2)
		img_coeffs_slices.append(temp3)

	img_coeffs_array = np.stack(img_coeffs_array, axis = 0)
	return img_coeffs, img_coeffs_array, img_coeffs_slices


def reconstruct_image_array(img_coefficients, wavelet='db1'):
	temp1 = []
	for i in range(3):
		temp2 = pywt.waverec2(img_coefficients[i], wavelet)
		temp1.append(temp2)

	image_recon_array = np.stack(temp1, axis=2)
	return image_recon_array
	

def SWA(array_coefficients, tc, perturbation='none'):
	temp = copy.copy(array_coefficients)

	if perturbation == 'none':
		print('Applying perturbation type: ' + perturbation)
		return temp

	elif perturbation == 'dumb':
		print('Applying perturbation type: ' + perturbation)
		alpha = 0
		print(len(tc))
		for i in range(len(tc)):
			print(temp[tc[i][0]][tc[i][1]][tc[i][2]])
			temp[tc[i][0]][tc[i][1]][tc[i][2]] = temp[tc[i][0]][tc[i][1]][tc[i][2]]*alpha
			print(temp[tc[i][0]][tc[i][1]][tc[i][2]])
		return temp
	else:
		print('Perturbation type not recognised, no perturbation applied - returing original coefficients')
		return temp


def get_target_coefficients(img_coeffs_array, K):
	coeff_size_order = np.dstack(np.unravel_index(np.argsort(img_coeffs_array.ravel()), (img_coeffs_array.shape[0], img_coeffs_array.shape[1], img_coeffs_array.shape[2])))
	target_coeffs = coeff_size_order[0][0:K]
	return target_coeffs



def get_array_from_image(img):
	temp =  np.asarray(img)
	temp = np.swapaxes(temp, 0, 2)
	img_array = np.swapaxes(temp, 1, 2)
	return img_array


def get_image_from_array(arr):
	img = arr.clip(0, 255)
	img = img.astype('uint8')
	img = im.fromarray(img)
	return img






#Define global parameters
wavelet = 'db1'
image_path = "butterfly.png"
K = 25


#Select SWA perturbation
# perturbation = 'none'
perturbation = 'dumb'


#Open image
img=im.open(image_path)


#Convert image to numpy array in format [channel][row][column]
img_array = get_array_from_image(img)


#Calculate the wavelet coefficients of the image from the image array
img_coeffs, img_coeffs_array, img_coeffs_slices = deconstruct_image_array(img_array, wavelet)


#Identify the locations of the K largest coefficients in the coefficient array
target_coeffs = get_target_coefficients(img_coeffs_array, K)
print(target_coeffs)


#Apply some adversarial function to manipulate the image wavelet coefficients
img_coeffs_array_adv = SWA(img_coeffs_array, target_coeffs, perturbation)


#Pack image coefficients back into tuple form
img_coeffs_adv = []
for i in range(3):
	temp = pywt.array_to_coeffs(img_coeffs_array_adv[i], img_coeffs_slices[i], output_format="wavedec2")
	img_coeffs_adv.append(temp)


# Reconstuct image array from perturbed wavelet coefficients stored in tuple form
img_recon_array = reconstruct_image_array(img_coeffs_adv, wavelet)


# Reconstuct image from array
img_recon = get_image_from_array(img_recon_array)


# #Plot to compare output
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_recon)
plt.show()


