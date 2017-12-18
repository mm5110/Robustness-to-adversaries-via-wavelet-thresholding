# Demonstrates the implementation of a wavelet decomposition and reconstruction filter in pytorch and tests that the reconstructed
# and original are classified as the same.


import pywt

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import copy


import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch import nn
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import Dataset, DataLoader

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# GLOBAL VARIABLES
wavelet = 'bior2.2'
levels = 4 # (5 is limit) Due to he fact we scale all images to be 3 x 256 x256
# K = 1000
channel_means = [0.485, 0.456, 0.406]
channel_stds = [0.229, 0.224, 0.225]

# K_max = 2000
# K_min = 25
# numb_tests = 100
# K_step = int((K_max-K_min)/numb_tests)

mask_image_enhance = 100
steps = 1000
eps = 2 * 8 / 225
step_alpha = 0.05

# img_path = 'butterfly.jpg'
img_path = 'car.png'
# img_path = 'panda.png'




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# PLOT SELECTED WAVELET
w=pywt.Wavelet('bior2.2')
# plt.plot(w.dec_hi[::-1], label="dec hi")
# plt.plot(w.dec_lo[::-1], label="dec lo")
# plt.plot(w.rec_hi, label="rec hi")
# plt.plot(w.rec_lo, label="rec lo")
# plt.title("Bior 2.2 Wavelets")
# plt.legend()
# plt.show()


# DEFINE WAVELET FILTERS AND WAVELET TRANSFORMS
dec_hi = torch.Tensor(w.dec_hi[::-1]) 
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0).double()

inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0).double()


# NOTE FOR FUTURE - WE DO NOT CURRENTLY HANDLE THE BORDER CASES WHICH LEADS TO ERRORS DURING RECONSTRUCTION
def wt(vimg, levels=1):
    h = vimg.size(2)
    w = vimg.size(3)
    padded = torch.nn.functional.pad(vimg,(2,2,2,2))
    res = torch.nn.functional.conv2d(padded, Variable(filters[:,None]),stride=2)
    if levels>1:
        res[:,:1] = wt(res[:,:1],levels-1)
    res = res.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)
    return res


def iwt(vres, levels=1):
    h = vres.size(2)
    w = vres.size(3)
    res = vres.view(-1,h//2,2,w//2).transpose(1,2).contiguous().view(-1,4,h//2,w//2).clone()
    if levels>1:
        res[:,:1] = iwt(res[:,:1], levels=levels-1)
    res = torch.nn.functional.conv_transpose2d(res, Variable(inv_filters[:,None]),stride=2)
    res = res[:,:,2:-2,2:-2]
    return res


def calc_wav_coeffs_tensor_from_image_tensor(vimg, levels=1):
    # Input: takes tensor of pixel values with dimensions as [batch][channel][height][width]
    # Output: returns tensor of wavelet coefficients with dimensions as [batch][channel][height][width]
    # A given layer will contain the 2D wavelet transform of a given 2D layer with coars information top left corner
    c = vimg.size(1)
    coeffs_list = []
    for i in range(c):
        layer = vimg[0][i]
        layer = layer[None][None]
        layer_coeffs = wt(layer, levels)
        coeffs_list.append(layer_coeffs)
    coeffs = torch.cat(coeffs_list, dim=1)
    coeffs = coeffs.double()
    return coeffs.data


def calc_image_var_from_coeffs_var(coeffs_var, levels=1):
    # Input: takes tensor of wavelet coefficients with dimensions as [batch][channel][height][width]
    # Output: returns tensor of pixel values with dimensions as [batch][channel][height][width]
    c = coeffs_var.size(1)
    image_list = []
    for i in range(c):
        layer = coeffs_var[0][i]
        layer = layer[None][None]
        layer_image = iwt(layer, levels)
        image_list.append(layer_image)
    image = torch.cat(image_list, dim=1)
    image = image.double()
    return image


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# IMAGE HANDLING FUNCTIONS AND TRANSFORMS

normalize = transforms.Normalize(
    mean=channel_means,
    std=channel_stds
)
preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

no_preprocess = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor()])

def unprocess_tensor(img_tensor):
  # temp = img_tensor.numpy()
  temp = copy.copy(img_tensor)
  for i in range(3):
    temp[0][i] = temp[0][i]*channel_stds[i] + channel_means[i]
  temp = temp*256
  return temp

def load_image(img_path):
  img_pil = Image.open(img_path).convert('RGB')
  img_tensor = no_preprocess(img_pil)*256
  img_tensor_norm = preprocess(img_pil)
  img_tensor_norm = img_tensor_norm.unsqueeze_(0)
  img_tensor.unsqueeze_(0) # This line is if want to process a batch of images in a 4d array/ tensor
  return img_pil, img_tensor.double(), img_tensor_norm.double()


def get_image_from_tensor(img_tensor):
  temp = img_tensor.numpy()
  temp = np.squeeze(temp, axis=0) 
  temp = np.swapaxes(temp, 0, 1)
  temp = np.swapaxes(temp, 1, 2)
  temp = temp.clip(0, 255)
  temp = temp.astype('uint8')
  img_pil = Image.fromarray(temp)
  return img_pil


def plot_images_sbs(original, mask, peturbed, original_class, peturbed_class, mask_image_enhance, figure_num=1):
  plt.figure(figure_num, figsize=(20,10))
  plt.subplot(1,3,1)
  plt.title('Original');
  plt.xlabel('Classification: ' + original_class);
  plt.imshow(original)
  plt.subplot(1,3,2)
  plt.title('Mask');
  plt.xlabel('Brightened by factor ' + str(mask_image_enhance));
  plt.imshow(mask)
  plt.subplot(1,3,3)
  plt.title('Perturbed')
  plt.xlabel('Classification: ' + peturbed_class);
  plt.imshow(peturbed)
  plt.show()
  return None 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     

# SWA ATTACK ALGORITHM
def get_K_target_coeff_indices(img_coeffs, K):
  # Inputs: the wavelet coefficient array of an image and K - the number of coefficients of interest
  # Output: locations of the K largest coefficients in the wavelet coefficient array as a torch tensor
  img_coeffs_array = img_coeffs.numpy()
  coeff_size_order = np.dstack(np.unravel_index(np.argsort(abs(img_coeffs_array).ravel()), (img_coeffs_array.shape[0], img_coeffs_array.shape[1], img_coeffs_array.shape[2], img_coeffs_array.shape[3])))
  target_coeffs = coeff_size_order[0][-K:len(coeff_size_order[0])]
  # target_coeffs = torch.from_numpy(target_coeffs)
  return target_coeffs


def create_adversary_nontargeted(original_img_tensor, levels, K, labels):

    # Calculate wavelet transform of image
    original_coeffs_tensor = calc_wav_coeffs_tensor_from_image_tensor(original_img_tensor, levels)

    # Find locations of K largest coefficients (target coefficients)
    tc = get_K_target_coeff_indices(original_coeffs_tensor, K)

    # Copy / clone the coefficients of the image to a new Variable
    y = Variable(torch.zeros(1, 1))
    perturbed_coeffs_var = Variable(original_coeffs_tensor.clone(), requires_grad = True)
    # Execute gradient descent attack on wavelet target wavelet coefficients:
    for step in range(steps):
      zero_gradients(perturbed_coeffs_var)
      perturbed_img_var = calc_image_var_from_coeffs_var(perturbed_coeffs_var, levels)
      out = model(perturbed_img_var)
      y.data = out.data.max(1)[1]
      classification = labels[out.data.numpy().argmax()]
      classification_confidence = out.max()
      if step == 0:
        original_classification = copy.copy(classification)
        original_confidence = out.max()
        print("\nClassification of original image: " + str(original_classification) + ' with confidence ' + str(original_confidence.data))

      if classification != original_classification:
        print("\nAttack successful at step " + str(step) + ": miss-classification detected - peturbed image now classified as: " + str(classification) + ' with confidence ' + str(classification_confidence.data))
        exit_status = 1
        break
      
      _loss = loss(out, y)
      _loss.backward()
 
      for i in range(len(tc)):
          perturbed_coeffs_var.data[tc[i][0]][tc[i][1]][tc[i][2]][tc[i][3]] = perturbed_coeffs_var.data[tc[i][0]][tc[i][1]][tc[i][2]][tc[i][3]] + step_alpha*(np.sign(perturbed_coeffs_var.grad.data[tc[i][0]][tc[i][1]][tc[i][2]][tc[i][3]]))

    if (step+1) == steps:
      print("\nAttack unsuccessful after " + str(steps) + " steps, consider increasing number of steps or increasing step size")
      exit_status = 0

    # Create image from perturbed coefficients:
    perturbed_img_var = calc_image_var_from_coeffs_var(perturbed_coeffs_var, levels)
    perturbed_img_tensor = perturbed_img_var.data 
    mask_img_tensor = perturbed_img_tensor - original_img_tensor

    for i in range(3):
      mask_img_tensor[0][i] = mask_img_tensor[0][i] - channel_means[i]/channel_stds[i]

    # Remove image normalisation from tensor
    perturbed_img_tensor = unprocess_tensor(perturbed_img_tensor)
    mask_img_tensor = unprocess_tensor(mask_img_tensor)
    mask_img_tensor_energy = torch.norm(mask_img_tensor,2)
    perturbed_img_tensor_energy = torch.norm(perturbed_img_tensor,2)
    energy_ratio = mask_img_tensor_energy/perturbed_img_tensor_energy

    # Create Pil images from tensors
    perturbed_img_pil = get_image_from_tensor(perturbed_img_tensor)
    mask_img_pil = get_image_from_tensor(mask_image_enhance*mask_img_tensor)

    # Return the image pertubed image, tensor and the mask image
    # return perturbed_img_pil, perturbed_img_tensor, mask_img_pil, mask_img_tensor, original_classification, original_confidence, classification, classification_confidence, exit_status
    return exit_status, energy_ratio

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     

# IMPORT NETWORK:
# Import a pre-trained squeezenet
model = models.squeezenet1_1(pretrained=True).double()
# Define the loss function as the cross entropy
loss = nn.CrossEntropyLoss()
# Sets the module into evaluation mode - set self.train(False)
model.eval();

# Classify an image using the defined classifier network / model
def classify_image(img_tensor, classifier, labels):
  output = classifier(Variable(img_tensor, requires_grad=True))
  classification = labels[output.data.numpy().argmax()]
  return output, classification



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# ACTUAL PROGRAM...

# LOAD CLASSES
labels = eval(open('classes.txt').read())

# LOAD IN IMAGE TO PROCESS
original_img_pil, original_img_tensor, original_img_tensor_norm = load_image(img_path)

# K_span = np.arange(K_min, K_max, K_step)

K_span = [25, 50, 100, 200, 300, 400, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000]
mask_energies = np.zeros(len(K_span))
attack_success_record = np.zeros(len(K_span))
i = 0
for iter in K_span:
	print("Generating adversary with sparsity = " + str(int(iter)))
	exit_status, energy_ratio = create_adversary_nontargeted(original_img_tensor_norm, levels, int(iter), labels)
	mask_energies[i] = energy_ratio
	attack_success_record[i] = exit_status
	i = i+1

data=np.stack((K_span, mask_energies, attack_success_record), axis=0)
np.save('mask_sparsity_vs_energy_2', data)


# colors =[]
# for i in range(len(K_span)):
# 	if attack_success_record[i] == 0:
# 		colors.append('red')
# 	else:
# 		colors.append('blue')


# plt.scatter(K_span, mask_energies*100, c=colors, cmap=matplotlib.colors.ListedColormap(colors))
# plt.suptitle('Energy vs sparsity of mask that resulted in misclassification', fontsize=14, fontweight='bold')
# plt.title('Levels = ' + str(levels) + ', Image = ' + str(img_path) + ', Step size = ' + str(step_alpha) + ', Max number of steps = ' + str(steps))
# plt.xlabel('Sparsity K')
# plt.ylabel('l2 Energy ratio Mask/Original (%)')
# plt.show()







