from scipy import misc
from PIL import Image as im

#im_path = ""

#Read in images both as a a png and a numpy array
panda_png = im.open("panda.png")
panda_arr = misc.imread("panda.png")

print(type(panda_png))
print(type(panda_arr))

#Execute basic functions on image
print("Image size is: " + str(panda_arr.shape) + ". Data type of is: " + str(panda_arr.dtype))