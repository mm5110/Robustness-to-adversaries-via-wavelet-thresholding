import pywt
import matplotlib.pyplot as plt
import numpy
from PIL import Image
import urllib.request
import io
import torch
from torch.autograd import Variable

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# GLOBAL VARIABLES
wavelet = 'bior2.2'
levels = 3

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


# NOTE FOR FUTURE WE DO NOT CURRENTLY HANDLE THE BORDER CASES WHICH LEADS TO ERRORS DURING RECONSTRUCTION
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


def calc_wav_coeffs_from_image_tensor(vimg, levels=1):
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
    return coeffs


def calc_image_tensor_from_wav_coeffs(coeffs, levels=1):
    # Input: takes tensor of wavelet coefficients with dimensions as [batch][channel][height][width]
    # Output: returns tensor of pixel values with dimensions as [batch][channel][height][width]
    c = coeffs.size(1)
    image_list = []
    for i in range(c):
        layer = coeffs[0][i]
        layer = layer[None][None]
        layer_image = iwt(layer, levels)
        image_list.append(layer_image)
    image = torch.cat(image_list, dim=1)
    return image


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

x = torch.rand(1,3,8,8).double()

vx = Variable(x, requires_grad=True)

vx_coeffs = calc_wav_coeffs_from_image_tensor(vx, levels)

vx_recon = calc_image_tensor_from_wav_coeffs(vx_coeffs, levels)

print(vx_recon - vx)





