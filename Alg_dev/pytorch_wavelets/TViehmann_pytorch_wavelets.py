import pywt
import matplotlib.pyplot as plt
# matplotlib inline
import numpy
from PIL import Image
import urllib.request
import io
import torch
from torch.autograd import Variable

URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Zuse-Z4-Totale_deutsches-museum.jpg/315px-Zuse-Z4-Totale_deutsches-museum.jpg'

print(pywt.families())

w=pywt.Wavelet('bior2.2')
# plt.plot(w.dec_hi[::-1], label="dec hi")
# plt.plot(w.dec_lo[::-1], label="dec lo")
# plt.plot(w.rec_hi, label="rec hi")
# plt.plot(w.rec_lo, label="rec lo")
# plt.title("Bior 2.2 Wavelets")
# plt.legend()
# plt.show()
dec_hi = torch.Tensor(w.dec_hi[::-1]) 
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

imgraw = Image.open(io.BytesIO(urllib.request.urlopen(URL).read())).resize((256,256))
img = numpy.array(imgraw).mean(2)/255
img = img.astype('float')
print(type(img))

print("'img' is dimension: ")
print(img.shape)

img = torch.from_numpy(img)
# plt.figure()
# plt.imshow(img, cmap=plt.cm.gray)

filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0).double()

inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0).double()



def wt(vimg, levels=1):
    print(vimg)
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


# img is
# print("'img' is: ")
# print(img)


# img[None, None] - the [None,None] args turn it into a 4D tensor 1x1x256x256
# print("'img[None, None]' is: ")
# print(img[None, None])

vimg = Variable(img[None,None])
print(vimg)

# Calculate wavelet transform, decompose into coefficients
res = wt(vimg,4)
plt.figure()
plt.imshow(res[0,0].data.numpy(),cmap=plt.cm.gray)
plt.show()

# Reconstruct image from wavelet transform
rec = iwt(res, levels=4)
plt.imshow(rec[0,0].data.numpy(),cmap=plt.cm.gray)
plt.show()

# Reconstruction error
plt.imshow((rec-vimg).data[0,0].numpy(), cmap=plt.cm.gray)
plt.colorbar()
plt.show()

