# import torch
# from torch import nn
# from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# from torchvision.models.inception import 

import io
import requests

import pywt

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torchvision.models.inception import inception_v3
from torch import nn
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import Dataset, DataLoader


# GLOBAL VARIABLES
wavelet = 'bior2.2'
levels = 3
# img_path = 'car.png'
img_path = 'butterfly.jpg'
eps = 2 * 8 / 225. 
steps = 50
norm = float('inf')
step_alpha = 0.01 
channel_means = [0.485, 0.456, 0.406]
channel_stds = [0.229, 0.224, 0.225]

# DEFINE IMAGE TRANSFORMS
# Standard normalisation and pre-process procesdures taken from Pytorch library to map betwen pytorch and imaagnetv3
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


# IMPORT NETWORK:
# Import a pre-trained squeezenet
model = models.squeezenet1_1(pretrained=True)
# Define the loss function as the cross entropy
loss = nn.CrossEntropyLoss()
# Sets the module into evaluation mode - set self.train(False)
model.eval();


# Setup wavelet stuff...
w=pywt.Wavelet(wavelet)
dec_hi = torch.Tensor(w.dec_hi[::-1]) 
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)


# DEFINE FUNCTIONS
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

def calculate_wavelet_coeffs(img_tensor, levels=1):
	print(img_tensor)
	img_tensor = torch.squeeze(img_tensor, dim=0)
	print(img_tensor)
	temp=[]
	for i in range(3):
		print(img_tensor[i])
		temp.append(wt_layer(img_tensor[i], levels))

	coeffs = torch.stack(temp, dim=0)
	print(coeffs)
	return coeffs


def load_image(img_path):
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0) # This line is if want to process a batch of images in a 4d array/ tensor
    return img_pil, img_tensor

def classify_image(img_tensor, classifier, labels):
	output = classifier(Variable(img_tensor, requires_grad=True))
	classification = labels[output.data.numpy().argmax()]
	return output, classification

def non_targeted_attack(img_tensor):
	label = torch.zeros(1, 1)
	x, y = Variable(img_tensor, requires_grad=True), Variable(label)
	for step in range(steps):
		zero_gradients(x)
		out = model(x)
		y.data = out.data.max(1)[1]
		_loss = loss(out, y)
		_loss.backward()
		step_adv = x.data + step_alpha * torch.sign(x.grad.data)
		adv = step_adv - img_tensor
		adv = torch.clamp(adv, -eps, eps)
		result = img_tensor + adv
		result = torch.clamp(result, 0.0, 1.0)
		x.data = result
	return result, adv


def SWA_non_targeted_attack(img_tensor):
	label = torch.zeros(1, 1)
	x, y = Variable(img_tensor, requires_grad=True), Variable(label)

	# Take wavelet transform, find K largest coefficients

	for step in range(steps):
		zero_gradients(x)
		out = model(x)
		y.data = out.data.max(1)[1]
		_loss = loss(out, y)
		_loss.backward()
		step_adv = x.data + step_alpha * torch.sign(x.grad.data)
		adv = step_adv - img_tensor
		adv = torch.clamp(adv, -eps, eps)
		result = img_tensor + adv
		result = torch.clamp(result, 0.0, 1.0)
		x.data = result
	return result, adv


def get_image_from_tensor(img_tensor):
	temp = img_tensor.numpy()
	temp = np.squeeze(temp, axis=0)
	for i in range(3):
		temp[i] = temp[i]*channel_stds[i] + channel_means[i]
		# temp[i] = temp[i]*channel_stds[i]
	temp = temp*255
	temp = np.swapaxes(temp, 0, 1)
	temp = np.swapaxes(temp, 1, 2)
	temp = temp.clip(0, 255)
	temp = temp.astype('uint8')
	img_pil = Image.fromarray(temp)
	return img_pil


def plot_images_sbs(original, mask, peturbed, original_class, peturbed_class, figure_num=1):
	plt.figure(figure_num, figsize=(20,10))
	plt.subplot(1,3,1)
	plt.title('Original');
	plt.xlabel('Classification: ' + original_class);
	plt.imshow(original)
	plt.subplot(1,3,2)
	plt.title('Mask');
	plt.imshow(mask)
	plt.subplot(1,3,3)
	plt.title('Perturbed')
	plt.xlabel('Classification: ' + peturbed_class);
	plt.imshow(peturbed)
	plt.show()
	return None


# CHECK Wavelet functions etc
x = torch.rand(1,3,5,5)

# x_coeffs = calculate_wavelet_coeffs(x, levels)

x_coeffs = wt(x, levels)


print(x_coeffs)






# LOAD CLASSES
# labels = eval(open('classes.txt').read())

# LOAD IN IMAGE TO PROCESS
# img_pil, img_tensor = load_image(img_path)


# output, classification = classify_image(img_tensor, model, labels)

# peturbed_img_tensor, mask_img_tensor = non_targeted_attack(img_tensor)


# peturbed_output, peturbed_classification = classify_image(peturbed_img_tensor, model, labels)
# print(peturbed_classification)

# peturbed_img_pil = get_image_from_tensor(peturbed_img_tensor)
# mask_img_pil = get_image_from_tensor(mask_img_tensor)

# plot_images_sbs(img_pil, mask_img_pil, peturbed_img_pil, classification, peturbed_classification, figure_num=1)


