# import torch
# from torch import nn
# from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# from torchvision.models.inception import 

import io
import requests

from PIL import Image

import matplotlib.pyplot as plt

from torchvision import models, transforms
from torch.autograd import Variable
from torchvision.models.inception import inception_v3
from torch import nn
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import Dataset, DataLoader


# GLOBAL VARIABLES
img_path = 'car.png'

# DEFINE IMAGE TRANSFORMS
# Standard normalisation and pre-process procesdures taken from Pytorch library to map betwen pytorch and imaagnetv3
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# DEFINE FUNCTIONS
def load_image(img_path):
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0) # This line is if want to process a batch of images in a 4d array/ tensor
    return img_pil, img_tensor

def get_class(img):
    x = Variable(img, volatile=True).cpu()
    cls = model(x).data.max(1)[1].cpu().numpy()[0]
    return classes[cls]

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

def classify_image(img_tensor_variable, classifier, labels):
	output = classifier(img_tensor_variable)
	classification = labels[output.data.numpy().argmax()]
	return output, classification





# IMPORT NETWORK:
# Import inctionv3 trained on Imagenet from Google (read https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html). 
# model = inception_v3(pretrained=True)
model = models.squeezenet1_1(pretrained=True)
# Define the loss function as the cross entropy
loss = nn.CrossEntropyLoss()
# Sets the module into evaluation mode - set self.train(False)
model.eval();




# LOAD CLASSES
labels = eval(open('classes.txt').read())
# LABELS_URL = 'http://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# labels = {int(key):value for (key, value) in requests.get(LABELS_URL).json().items()}

# print(type(labels))
# print(labels)
# print(labels[0])

# LOAD IN IMAGE TO PROCESS
img_pil, img_tensor = load_image(img_path)

output = model(Variable(img_tensor, requires_grad=True))
classification = labels[output.data.numpy().argmax()]
print(classification)
# output, classification = classify_image(Variable(img_tensor), model, labels)






# plot_images_sbs(img_pil, img_pil, img_pil, figure_num=1)


