# Intro:
# Purpose of this script is to demonstrate the ability to access the key wavelet coefficients of an image and apply some basic perturbation to the K most significant wavelets.
# When plotting the outcome of this so coined "Sparse Wavelet Attack" it the perturbation of the selected dumb attack should be clear.


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#Import libraries
import pywt
from scipy import misc
from scipy import fftpack
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np
import copy

from classify_image import run_inference_on_image

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#Define Functions
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
		for i in range(len(tc)):
			temp[tc[i][0]][tc[i][1]][tc[i][2]] = temp[tc[i][0]][tc[i][1]][tc[i][2]]*(5*np.random.uniform(0,1)+0.1)
		return temp
	else:
		print('Perturbation type not recognised, no perturbation applied - returning original coefficients')
		return temp

def get_image_array_from_coeffs(img_coeffs_array, img_coeffs_slices, wavelet):
	# Input: image_coefficient array, the image coefficient slices and the wavelet type
	# Output: reconstructed image array
	temp_layers = []

	for i in range(3):
		temp_tuple = pywt.array_to_coeffs(img_coeffs_array[i], img_coeffs_slices[i], output_format="wavedec2")
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


def plot_images_sbs(img1, img2, img3, figure_num=1):
	plt.figure(figure_num, figsize=(20,10))
	plt.subplot(1,3,1)
	plt.title('Original');
	plt.imshow(img1)
	plt.subplot(1,3,2)
	plt.title('Mask');
	plt.imshow(img2)
	plt.subplot(1,3,3)
	plt.title('Perturbed')
	plt.imshow(img3)
	plt.show()
	return None	

def plot_comparison_of_coeffs_dist(coefficients1, coefficients2, figure_numb=1, min_coefficient_val=1):	
	#PLOT DISTRIBUTION OF COEFFICIENTS in figure 1
	#Plot log distributions of coefficients
	plt.figure(figure_numb, figsize=(30,10))
	plt.subplot(221)
	plt.scatter(range(coefficients1[abs(coefficients1)>min_coefficient_val].ravel().size), np.log10(np.abs(coefficients1[abs(coefficients1)>min_coefficient_val].ravel())), c='#348ABD', alpha=.3);
	plt.title('Clean');
	plt.xlabel('Order of wavelet Coefficients');
	plt.ylabel('Wavelet Coefficients Amplitude in log scale');
	plt.figure(figure_numb)
	plt.subplot(222)
	plt.scatter(range(coefficients2[abs(coefficients2)>min_coefficient_val].ravel().size), np.log10(np.abs(coefficients2[abs(coefficients2)>min_coefficient_val].ravel())), c='#348ABD', alpha=.3);
	plt.title('Perturbed');
	plt.xlabel('Order of wavelet Coefficients');
	plt.ylabel('Wavelet Coefficients Amplitude in log scale');
	#Plot Histrograms
	plt.subplot(223)
	plt.hist(np.log10(np.abs(coefficients1[abs(coefficients1)>min_coefficient_val].ravel())), bins=100, color='#348ABD', alpha=.3, histtype='stepfilled');
	plt.xlabel('Amplitude of Wavelet Coefficients (log-scaled)');
	plt.ylabel('Number of Wavelet Coefficients');
	plt.title('Clean');
	plt.subplot(224)
	plt.hist(np.log10(np.abs(coefficients2[abs(coefficients2)>min_coefficient_val].ravel())), bins=100, color='#348ABD', alpha=.3, histtype='stepfilled');
	plt.xlabel('Amplitude of Wavelet Coefficients (log-scaled)');
	plt.ylabel('Number of Wavelet Coefficients');
	plt.title('Perturbed');
	plt.show()
	return


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


#Define global parameters
wavelet = 'db4'
image_path = "butterfly.jpg"
min_coefficient_val = 10
K = 40


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# PROGRAM
#Select SWA perturbation
# perturbation = 'none'
perturbation = 'dumb'


#Open image
img=im.open(image_path)


#Convert image to numpy array in format [channel][row][column]
img_array = get_array_from_image(img)
# print(img_array[0][50][50])
# print(img_array[:][200][200])


#Calculate the wavelet coefficients of the image from the image array
img_coeffs_array, img_coeffs_slices = get_image_coeff_array(img_array, wavelet)
img_array_reconstructed = get_image_array_from_coeffs(img_coeffs_array, img_coeffs_slices, wavelet)

#Identify the locations of the K largest coefficients in the coefficient array
target_coeffs = get_K_target_coeff_indices(img_coeffs_array, K)


#Apply some adversarial function to manipulate the image wavelet coefficients
img_coeffs_array_adv = get_swa_coeff_array(img_coeffs_array, target_coeffs, perturbation)


# Reconstruct image arrays
img_array_perturbed = get_image_array_from_coeffs(img_coeffs_array_adv, img_coeffs_slices, wavelet)
img_array_mask = np.add(img_array_perturbed, (-1)*img_array_reconstructed)


# Reconstuct image from array and reconstruct mask image
img_perturbed = get_image_from_array(img_array_perturbed)
img_reconstructed = get_image_from_array(img_array_reconstructed)
img_mask = get_image_from_array(img_array_mask)


# print(img_array_perturbed[:][50][50])
# print(img_array_perturbed[:][200][200])



# Plot to compare output
print("Displaying image, image mask and perturbed image... ")
plot_images_sbs(img_reconstructed, img_mask, img_perturbed, 1)


print("Plotting coefficients ... ")
plot_comparison_of_coeffs_dist(img_coeffs_array, img_coeffs_array_adv, 2)

img_perturbed.save("butterfly_peturbed.jpg")
print("Saving image as jpg: butterfly_peturbed.jpg")




