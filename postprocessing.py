import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from numba import njit, prange
from torch import nn
import SimpleITK as sitk
sys.path.append('..')
from utils.maskprocess import boundary_mask

@njit(parallel=True,fastmath=True, nogil=True)
def min_brainsize(img,min_brain_vox = 5000):
	out = np.zeros(img.shape[0])
	for i in prange(img.shape[0]):
		cond = (img[i]<80) & (img[i]>0)
		out[i] = (cond.sum()>min_brain_vox)
	return out==1

@njit(parallel=True,fastmath=True, nogil=True)
def min_brainsize_withlabel(img,labels,min_brain_vox = 5000):
	out = np.zeros(img.shape[0])
	for i in range(img.shape[0]):
		cond = (img[i]<80) & (img[i]>0)
		out[i] = (cond.sum()>min_brain_vox)+(labels[i])
	return out>0

@njit(parallel=True,fastmath=True, nogil=True)
def remove_empty_slices(CT, minpixvals=-800, margin=1):
	"""
	Removes slices with a sum of pixel values <= maxpixval
	function uses numba njit for faster execution (speed x5)
	
	Input:
	CT: 3d np array with dimensions (x,y,z)
	maxpixval: maximum summed value of slice
	margin: selects n slices under/above the cut slices
	return_upper_lower: if the lower-upper bounds of the 
	original CT should be returned
	"""
	# lower indicates if lower bound of brain has been passed
	# upper indicates if the upper bound of brain has been passed
	lower, upper = False, False
	z_dim_CT = CT.shape[0]
	for i in range(0,z_dim_CT):
		sl = CT[i,:,:]
		pix_max = sl.max()
		# to select lower bound
		if (pix_max>minpixvals) & (lower==False):
			lower=True 
			lower_ix = i-margin
		# to select upper bound
		if (pix_max<=minpixvals) & (lower==True) & (upper==False):
			upper = True
			upper_ix = i+margin+1
			if upper_ix>z_dim_CT:
				upper_ix = z_dim_CT
				
	CT_out = CT[lower_ix:upper_ix,:,:]

	return CT_out, lower_ix, upper_ix

# function uses np not sitk
def apply_mask(m,img, foreground_m=1, background=-1024):
	if isinstance(m, sitk.SimpleITK.Image):
		m = sitk.GetArrayFromImage(m)
	if isinstance(img, sitk.SimpleITK.Image):
		img = sitk.GetArrayFromImage(img)
	img[m==foreground_m]=background
	return img

def sitk_apply_mask(m,img, foreground_m=0, background=None, sitk_type = sitk.sitkInt32):
	"""
	Applies mask (m) to an image, sets background of image
	returns and image with only mask foreground
	"""
	m = sitk.Cast(m,sitk_type)
	img = sitk.Cast(img,sitk_type)
	if foreground_m==0:
		mf = sitk.MaskNegatedImageFilter()
	elif foreground_m==1:
		mf = sitk.MaskImageFilter()
	if background!=None:
		mf.SetOutsideValue(background)
	return mf.Execute(img, m)

def dict_crop2D(d,img):
	"""
	Uses a dictionary with upper lower
	boundaries defined as dimdict
	in maskprocess.boundary_mask
	d:dictionary, img: np.ndarray
	"""
	return img[d[0]['lower']:d[0]['upper'],d[1]['lower']:d[1]['upper']]

def local_mask(mask,margin):
	# computes a square mask surrounding the foreground
	# adds a margin around min and max indices of mask
	bd = boundary_mask(mask,dims=[0,1,2],foregroundval=1)
	m2 = np.zeros_like(mask)
	m2[bd[0]['lower']-margin:bd[0]['upper']+margin,
	   bd[1]['lower']-margin:bd[1]['upper']+margin,
	   bd[2]['lower']-margin:bd[2]['upper']+margin] = 1
	return m2

def crop_boundary_dct(img,dct, margin):
	"""
	img (np) is cropped based on a dictionary (dct)
	in which per dimension lower-upper bound
	of the foreground are defined
	margin is used to add extra space around
	the foreground
	"""
	
	if isinstance(margin,int):
		m0 = m1 = m2 = margin
	elif isinstance(margin, float):
		d0 = abs(dct[0]['upper']-dct[0]['lower'])
		d1 = abs(dct[1]['upper']-dct[1]['lower'])
		d2 = abs(dct[2]['upper']-dct[2]['lower'])
		m0,m1,m2 = margin*d0,margin*d1,margin*d2
	elif isinstance(margin,tuple) or isinstance(margin,list):
		m0,m1,m2 = margin

	if isinstance(img,np.ndarray):
		s0,s1,s2 = img.shape # total dim lengths
		i0,i1,i2 = 0,1,2 # indices for dct
	elif isinstance(img,sitk.SimpleITK.Image):
		s2,s1,s0 = img.GetSize()
		i0,i1,i2 = 0,1,2 # itk images have different arangement     
		
	dzL, dzH = max(0,dct[i0]['lower']-m0),min(s0,dct[i0]['upper']+m0)
	dyL, dyH = max(0,dct[i1]['lower']-m1),min(s1,dct[i1]['upper']+m1)
	dxL, dxH = max(0,dct[i2]['lower']-m2),min(s2,dct[i2]['upper']+m2)
	
	if isinstance(img,np.ndarray):
		return img[int(dzL):int(dzH), int(dyL):int(dyH), int(dxL):int(dxH)]
	elif isinstance(img,sitk.SimpleITK.Image):
		return img[int(dxL):int(dxH), int(dyL):int(dyH), int(dzL):int(dzH)]


def mask_overlap_zcut(m1,m2,imgs, visualize=True, txt='', max_delta=30000, min_p_slice=.4):
	# sometimes NCCT is cut higher (in z) than CTA, if this is the case remove slices below this level
	# Find where the overlap of head masks starts (otherwise NCCT-CTA info does not match)
	good_overl = False
	for i_overl in range(0,m1.shape[0]):
		if m1[i_overl].sum()<5000 or m2[i_overl].sum()<5000: 
			continue
		delta_hm = abs(m1[i_overl]-m2[i_overl]).sum()
		if delta_hm<max_delta: # non overlapping pixels should not be larger than threshold
			good_overl = True
			break
	if i_overl>int(m1.shape[0]*min_p_slice):
		print(txt, 'too high min index:', i_overl, m1.shape[0])
		good_overl = False
		
	if good_overl:
		if visualize:
			plt.imshow(np.hstack([img[i_overl] for img in imgs]))
			plt.axis('off')
			plt.title(txt+' good overlap')
			plt.show()
		
		imgs_out = [img[i_overl:] for img in imgs]

	else:
		i = int(m1.shape[0]*.8)
		if visualize:
			plt.imshow(np.hstack([img[i] for img in imgs]))
			plt.axis('off')
			plt.title(txt + ' poor overlap')
			plt.show()
		imgs_out = imgs
		i_overl = 0
	return imgs_out, i_overl


def MultipleMorphology(mask,
					   operations,
					   mm_rads,
					   foreground=1):
	"""
	Consecutively performs multiple morphology operations
	"""
	if len(operations) != len(mm_rads):
		print('Error: number of operations is not equal to number of radius (mm_rads)')

	rads_3d = []
	for rad in mm_rads:
		tmp = (int(math.floor(rad / mask.GetSpacing()[0])),
			   int(math.floor(rad / mask.GetSpacing()[1])),
			   int(math.floor(rad / mask.GetSpacing()[2])))
		rads_3d.append(tmp)

	for r, oper in zip(rads_3d, operations):
		oper.SetBackgroundValue(abs(foreground - 1))
		oper.SetForegroundValue(foreground)
		oper.SetKernelRadius(r)
		mask = oper.Execute(mask)

	return mask


class Morphology(nn.Module):
	"""
	Pytorch implementation:
	Class performs erosion (iters<0) or dilation (iters>0) for a specific
	number of iterations on the foreground mask of an image.
	The main benefit is that these operations can be performed on a
	GPU during training if device is set to 'cuda'. This results
	in up to 20x (10x more likely) faster mask computation. On 'cpu' this operation is
	slower than multiple scipy erosion or dilation steps.
	"""

	def __init__(self, iters=35, background=-1,
				 connect=8, type=torch.float32,
				 device='cuda', combine='OR', dim3D=False):
		super(Morphology, self).__init__()

		self.dim3D = dim3D
		if iters >= 0:
			self.operation = 'dilate'
			self.iters = iters
		else:
			self.operation = 'erode'
			self.iters = abs(iters)

		self.background = background
		self.type = type
		self.device = device
		self.connectivity = connect
		self.combine = combine  # how to combine multiple images into a mask

		if connect == 8:
			if not self.dim3D:
				self.kernel = torch.tensor([
					[1, 1, 1],
					[1, 1, 1],
					[1, 1, 1]],
					dtype=self.type
				).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3, 3) = (Batch,Channel,H,W)
			else:
				self.kernel = torch.ones([3, 3, 3], dtype=self.type).unsqueeze(0).unsqueeze(
					0)  # shape: (1, 1, 3, 3, 3) = (Batch,Channel,D,H,W)

		elif connect == 4:
			if not self.dim3D:
				self.kernel = torch.tensor([
					[0, 1, 0],
					[1, 1, 1],
					[0, 1, 0]],
					dtype=self.type
				).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3, 3) = (Batch,Channel,H,W)
			else:
				self.kernel = torch.tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
											[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
											[[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype=self.type
										   ).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3, 3,3) = (Batch,Channel,D,H,W)
		else:
			kernel = None

		if not self.dim3D:
			conv = nn.Conv2d(1, 1, kernel_size=self.kernel.shape[-1],
							 stride=1, padding=1, bias=False)
		else:
			conv = nn.Conv3d(1, 1, kernel_size=self.kernel.shape[-1],
							 stride=1, padding=1, bias=False)

		with torch.no_grad():
			conv.weight = nn.Parameter(self.kernel)
		self.conv = conv.type(self.type).to(self.device)

	# pass only img if on same device
	def __call__(self, img):
		# first construct the mask (foreground=1) from the image
		mask = (img > self.background).type(self.type)
		if self.iters > 0:
			# repeatedly erode or dilate
			for i in range(self.iters):
				if self.operation == 'dilate':
					mask = (self.conv(mask) > 0).type(self.type)
				if self.operation == 'erode':
					mask = (self.conv(mask) >= self.connectivity).type(self.type)
		# final threshold after erosion or dilation
		return mask

