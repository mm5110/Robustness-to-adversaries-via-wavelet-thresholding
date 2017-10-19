from scipy import misc
from scipy import fftpack
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np
import copy

def calc_2d_dct_image(img):
	print ("Calculating DCT coefficients of image...")
	coefficients = fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')
	print ("Coefficients calculated")
	return coefficients

def calc_2d_idct_image(coefficients):
	print ("Calculating image from DCT components")
	return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

def adv_simple(coefficients):
	temp = copy.copy(coefficients)
	temp[0,:,:] = 0
	return temp


def get_image_from_array(raw):
    img = raw.clip(0, 255)
    img = raw.astype('uint8')
    img = im.fromarray(img)
    return img


def plot_dct_coeffs_mat(coefficients):	
	#Viusualise distribution of coefficients using matrix
	plt.figure(1)
	plt.subplot(131)
	plt.imshow(np.abs(coefficients[:25, :25, 0]), cmap=plt.cm.Paired);
	plt.title('Red - First 100 coefficients in Grid')
	plt.subplot(132)
	plt.imshow(np.abs(coefficients[:25, :25, 1]), cmap=plt.cm.Paired);
	plt.title('Blue - First 100 coefficients in Grid')
	plt.subplot(133)
	plt.imshow(np.abs(coefficients[:25, :25, 2]), cmap=plt.cm.Paired);
	plt.title('Green - First 100 coefficients in Grid')
	plt.colorbar()
	plt.show()	
	return


def plot_dct_coeffs_dist(coefficients):	
	#PLOT DISTRIBUTION OF COEFFICIENTS in figure 1
	#Plot log distribution of coefficients
	plt.figure(1)
	plt.subplot(121)
	plt.scatter(range(coefficients.ravel().size), np.log10(np.abs(coefficients.ravel())), c='#348ABD', alpha=.3);
	plt.title('DCT Coefficient Amplitude vs. Order of Coefficient');
	plt.xlabel('Order of DCT Coefficients');
	plt.ylabel('DCT Coefficients Amplitude in log scale');
	#Plot Histrogram
	plt.subplot(122)
	plt.hist(np.log10(np.abs(coefficients.ravel())), bins=100, color='#348ABD', alpha=.3, histtype='stepfilled');
	plt.xlabel('Amplitude of DCT Coefficients (log-scaled)');
	plt.ylabel('Number of DCT Coefficients');
	plt.title('Amplitude distribution of DCT Coefficients');
	plt.show()
	return


def plot_compare_dct_coeffs_mat(coefficients1, coefficients2):	
	#Viusualise distribution of coefficients using matrix
	plt.figure(4)
	plt.subplot(231)
	plt.imshow(np.abs(coefficients1[:25, :25, 0]), cmap=plt.cm.Paired);
	plt.title('Red - clean')
	plt.subplot(232)
	plt.imshow(np.abs(coefficients1[:25, :25, 1]), cmap=plt.cm.Paired);
	plt.title('Blue - clean')
	plt.subplot(233)
	plt.imshow(np.abs(coefficients1[:25, :25, 2]), cmap=plt.cm.Paired);
	plt.title('Green - clean')
	plt.subplot(234)
	plt.imshow(np.abs(coefficients2[:25, :25, 0]), cmap=plt.cm.Paired);
	plt.title('Red - adjusted')
	plt.subplot(235)
	plt.imshow(np.abs(coefficients2[:25, :25, 1]), cmap=plt.cm.Paired);
	plt.title('Blue - adjusted')
	plt.subplot(236)
	plt.imshow(np.abs(coefficients2[:25, :25, 2]), cmap=plt.cm.Paired);
	plt.title('Green - adjusted')
	plt.colorbar()
	plt.show()	
	return



#Read in image as a numpy array
panda_clean = misc.imread("panda.png")

#Gather basic facts on image
print("Image size is: " + str(panda_clean.shape) + ". Data type of is: " + str(panda_clean.dtype))

#Calculate DCT
panda_dct_coeffs = calc_2d_dct_image(panda_clean)
print("DCT size is: " + str(panda_dct_coeffs.shape) + ". Data type of is: " + str(panda_dct_coeffs.dtype))

test = get_image_from_array(calc_2d_idct_image(panda_dct_coeffs))
plt.imshow(test)
plt.show()

#Apply some changes to DCT coefficients
panda_dct_coeffs_adj = adv_simple(panda_dct_coeffs)

test = get_image_from_array(calc_2d_idct_image(panda_dct_coeffs))
plt.imshow(test)
plt.show()


#Compare dct coefficients before and after adjustment
plot_compare_dct_coeffs_mat(panda_dct_coeffs, panda_dct_coeffs_adj)

#Reconstruct panda image from adjusted DCT coeffs
panda_recon = get_image_from_array(calc_2d_idct_image(panda_dct_coeffs_adj))

#Plot panda original vs reconstructed to observe visual differences in images
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(panda_clean)
plt.subplot(1,2,2)
plt.imshow(panda_recon)
plt.show()




