import numpy as np
import sys
from torch import nn
import torch
from typing import List

import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
sys.path.append('..')
from utils.preprocess import np2sitk, Resample_img
from utils.utils import make_odd, is_odd

def rtrn_np(img): # returns numpy array from torch tensor (on cuda)
	return img.detach().cpu().numpy()


def gauss_filter_downsample(img, new_spacing, device='cuda', ttype=torch.float32, max_ksize=None):
	# alternative: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/b75721a121102cf972f942fad927751089a7cc80/Python/05_Results_Visualization.ipynb
	# get 3d spacing to go to
	if isinstance(new_spacing, int) or isinstance(new_spacing, float):
		new_spacing = [new_spacing, new_spacing, new_spacing]

	# invert spacing for numpy array style
	NS = [new_spacing[2], new_spacing[1], new_spacing[0]]
	x, y, z = list(img.GetSpacing())
	SP = [z, y, x]
	vox_red = []  # kernel size that can be used for slice reduction
	mid = []  # middle index of kernel
	for sp, ns in zip(SP, NS):  # spacing, new spacing
		size = round(ns / sp)
		# get odd number for size
		if not is_odd(size):
			size = make_odd(size)
		size = max(size, 1)
		use_size = int(size)
		if max_ksize is not None:
			use_size = min(use_size, max_ksize)

		vox_red.append(use_size)
		mid.append(int(np.floor(size / 2)))

	# create gaussian kernel for convolution
	n = np.zeros(vox_red)
	n[mid[0], mid[1], mid[2]] = 1
	kernel = gaussian_filter(n, sigma=np.array(vox_red) / 6, order=0, mode='constant', cval=0)
	kernel = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
	# normalize kernel to sum up to 1
	kernel = (kernel * (1 / kernel.sum()))
	# use torch for conv since it can be done on gpu
	conv = nn.Conv3d(1, 1, kernel_size=vox_red, stride=1, padding='same', bias=False)
	conv.weight = nn.Parameter(kernel)
	conv = conv.type(ttype).to(device)
	for param in conv.parameters():
		param.requires_grad = False

	# use input_img
	input_img = sitk.GetArrayFromImage(img)
	mn_hu = input_img.min()
	if mn_hu < 0:  # add min (conv can not handle very negative values properly)
		input_img += mn_hu * -1
	input_img = torch.Tensor(input_img).unsqueeze(0).unsqueeze(0).type(ttype).to(device)
	new_img = rtrn_np(conv(input_img))[0, 0]
	if mn_hu < 0:  # subtract min again
		new_img += mn_hu
	new_img = np2sitk(new_img, img)
	return Resample_img(new_img, new_spacing=new_spacing, interpolator=sitk.sitkBSpline)