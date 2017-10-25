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

# NEW functions
def get_image_coeff_array(img_array, wavelet):
	# Input: the numpy array of an image and the wavelet type
	# Output: the wavelet coefficient array and slice tuple of the image
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
	return img_coeffs_array, img_coeffs_slices

def get_K_target_coeff_indices(img_coeffs_array, K):
	# Inputs: the wavelet coefficient array of an image and K - the number of coefficients of interest
	# Output: locations of the K largest coefficients in the wavelet coefficient array
	coeff_size_order = np.dstack(np.unravel_index(np.argsort(img_coeffs_array.ravel()), (img_coeffs_array.shape[0], img_coeffs_array.shape[1], img_coeffs_array.shape[2])))
	target_coeffs = coeff_size_order[0][0:K]
	return target_coeffs

def get_swa_coeff_array(img_coeffs_array, tc, perturbation='none'):
	# Input: the wavelet coefficient array of an image, the locations of the target indices and the perturbation to apply
	# Output: a peturbed coefficient array
	temp = copy.copy(img_coeffs_array)

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
		print('Perturbation type not recognised, no perturbation applied - returning original coefficients')
		return temp

def get_image_array_from_coeffs(img_coeffs_array, img_coeffs_slices, wavelet):
	# Input: image_coefficient array, the image coefficient slices and the wavelet type
	# Output: reconstructed image array
	temp_layers = []

	for i in range(3):
		temp_tuple = pywt.array_to_coeffs(img_coeffs_array_adv[i], img_coeffs_slices[i], output_format="wavedec2")
		temp_layers.append(pywt.waverec2(temp_tuple, wavelet))

	image_array = np.stack(temp_layers, axis=0)
	return image_array

def get_array_from_image(img):
	# Input: an image
	# Output: array of image data with dimensions aligned in format [d][r][c]
	temp =  np.asarray(img)
	temp = np.swapaxes(temp, 0, 2)
	img_array = np.swapaxes(temp, 1, 2)
	return img_array


def get_image_from_array(img_array):
	# Input: the numpy array of an image
	# Output: an image
	temp = np.swapaxes(img_array, 0, 1)
	temp = np.swapaxes(temp, 1, 2)
	temp = temp.clip(0, 255)
	temp = temp.astype('uint8')
	img = im.fromarray(temp)
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


