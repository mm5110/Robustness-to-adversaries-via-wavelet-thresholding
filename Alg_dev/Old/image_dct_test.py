from scipy import misc
from scipy import fftpack
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np

def calc_2d_dct_image(img):
	print ("Calculating DCT coefficients of image...")
	coefficients = fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')
	print ("Coefficients calculated")
	return coefficients

def calc_2d_idct_image(coefficients):
	print ("Calculating image from DCT components")
	return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')


def get_image_from_array(raw):
    img = raw.clip(0, 255)
    img = raw.astype('uint8')
    img = im.fromarray(img)
    return img


def plot_dct_coeffs_mat(coefficients):	
	#Viusualise distribution of coefficients using matrix
	plt.figure(1)
	plt.colorbar()	
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



#Read in image as a numpy array
panda_clean = misc.imread("panda.png")

#Gather basic facts on image
print("Image size is: " + str(panda_clean.shape) + ". Data type of is: " + str(panda_clean.dtype))

#Calculate DCT
panda_dct_coeffs = calc_2d_dct_image(panda_clean)
print("DCT size is: " + str(panda_dct_coeffs.shape) + ". Data type of is: " + str(panda_dct_coeffs.dtype))

#Plot dct coefficients
# plot_dct_coeffs_mat(panda_dct_coeffs)

panda_recon = get_image_from_array(calc_2d_idct_image(panda_dct_coeffs))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(panda_clean)
plt.subplot(1,2,2)
plt.imshow(panda_recon)
plt.show()



