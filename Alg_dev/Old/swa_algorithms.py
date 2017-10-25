import pywt
from scipy import misc
from scipy import fftpack
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np
import copy



filter_groups = ['cA', 'cH', 'cV', 'cD']

class ImWavTree_entry:
	def __init__(self, position, colour, colour_ind, level, filter_group, filter_group_ind, index, value, rank=0):
		self.position = position
		self.colour = colour
		self.colour_ind = colour_ind
		self.level = level
		self.filter_group = filter_group
		self.filter_group_ind = filter_group_ind
		self.index = index
		self.value = value
		self.rank = rank


class WavTree_list:
	def __init__(self):
		self.entries = []
		self.numb_entries = 0
		self.K = 1
	
	def add_entry(self, entry):
		self.entries.append(entry)

	# def identify_K_largest(self)

def unpack_w


def unpack_WavTree(WavTree, WavTree_list)

	



x = np.random.rand(5,5)
x_wc = pywt.wavedec2(x, 'db1')

colour = "Grey"
level = 1
filter_group = filter_groups[0]
index = 0

num_levels = len(x_wc)

for item in x_wc
	