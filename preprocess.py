import numpy as np
import os
import torch
from torch import nn
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
from utils.utils import rtrn_np, is_odd, make_odd

def dcm2sitk(input):
	#http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/03_Image_Details.html
	# creates an image from input
	# input can be a directory path
	# or a list of dicom file paths
	reader = sitk.ImageSeriesReader()
	if isinstance(input, str):
		dicom_names = reader.GetGDCMSeriesFileNames(input)
	elif isinstance(input, list):
		dicom_names = input
	else:
		print('Input is not appropriate for identifying dicoms:', input)
	reader.SetFileNames(dicom_names)
	reader.MetaDataDictionaryArrayUpdateOn()
	reader.LoadPrivateTagsOn()
	image = reader.Execute()
	return image

def dcm2niix(dcm2niix_exe, filename, output_dir, input_dir):
	#uses an executable to convort dicom to nifti
	#!activate root
	command = dcm2niix_exe + " -f "+filename+" -p y -z y -o"+ ' "'+ output_dir + '" "' + input_dir+ '"'
	os.system(command)

def assert_resliced_or_tilted(path,scanname='NCCT', ID=''):
	resl_tilted = [os.path.join(path,f) for f in os.listdir(path) \
				   if ('tilt' in f.lower() or 'eq' in f.lower()) and scanname.lower() in f.lower()]
	if len(resl_tilted)>0:
		p_ncct = resl_tilted [0]
		print(ID, scanname,'tilted or resliced:', '\n', p_ncct, '\n n adjusted:',len(resl_tilted))
		adjusted = True
	else:
		p_ncct = os.path.join(path,scanname+'.nii.gz')
		adjusted = False
	return p_ncct, adjusted

def sitk_flip_AP(img: sitk.SimpleITK.Image):
	return sitk.Flip(img, [False, True, False])

def np2sitk(arr: np.ndarray, original_img: sitk.SimpleITK.Image):
	img = sitk.GetImageFromArray(arr)
	img.SetSpacing(original_img.GetSpacing())
	img.SetOrigin(original_img.GetOrigin())
	img.SetDirection(original_img.GetDirection())
	# this does not allow cropping (such as removing thorax, neck)
	#img.CopyInformation(original_img) 
	return img

def RescaleInterceptHU(img):
	# if RescaleIntercept >-100 perform this transformation to get appropriate ranges
	img+=1024
	px_mode = 3096+1024
	img[img>=px_mode] = img[img>=px_mode] - px_mode
	img-=1024
	return img

def set_background(arr, mask, background=-1024):
	out = arr[mask==0] = background
	return out

def new_img_size(img,new_spacing):
	if isinstance(new_spacing,int) or isinstance(new_spacing,float):
		new_spacing = [new_spacing,new_spacing,new_spacing]
	new_size = []
	for ix,sp in enumerate(new_spacing):
		new_size.append(int(np.ceil(img.GetSize()[ix]*img.GetSpacing()[ix]/sp)))
	return new_size

def Resample_img(img, new_spacing=0.45, interpolator=sitk.sitkBSplineResamplerOrder3):
	# new_spacing should be in sitk order x,y,z (np order: z,y,x)
	if isinstance(new_spacing,int) or isinstance(new_spacing,float):
		new_spacing = [new_spacing,new_spacing,new_spacing]
	#https://github.com/SimpleITK/SimpleITK/issues/561
	resample = sitk.ResampleImageFilter()
	resample.SetInterpolator = interpolator
	resample.SetOutputDirection(img.GetDirection()) 
	resample.SetOutputOrigin(img.GetOrigin())
	resample.SetOutputSpacing(new_spacing)
	new_size = new_img_size(img,new_spacing)
	resample.SetSize(new_size)
	img = resample.Execute(img)
	img = sitk.Cast(img, sitk.sitkInt32)
	return img

def Resample_slices(img, new_z_spacing=5, interpolator=sitk.sitkBSplineResamplerOrder3):
	#https://github.com/SimpleITK/SimpleITK/issues/561
	resample = sitk.ResampleImageFilter()
	resample.SetInterpolator = interpolator
	resample.SetOutputDirection(img.GetDirection()) 
	resample.SetOutputOrigin(img.GetOrigin())
	new_spacing = [*img.GetSpacing()[:2],new_z_spacing]
	resample.SetOutputSpacing(new_spacing)
	new_n_slices = int(np.ceil(img.GetSize()[-1]*img.GetSpacing()[-1]/new_z_spacing))
	resample.SetSize((*img.GetSize()[:2], new_n_slices))
	img = resample.Execute(img)
	img = sitk.Cast(img, sitk.sitkInt32)
	return img

def Joint_Resample_img(ref_img,imgs, new_z_spacing=5, interpolator=sitk.sitkBSpline):
	#https://github.com/SimpleITK/SimpleITK/issues/561
	# function to joint undersample image and label (or 2 images)
	# images should have same origin,spacing,direction
	# interpolator options: sitk.sitkLinear, sitk.sitkBSpline, sitk.sitkNearestNeighbor
	resample = sitk.ResampleImageFilter()
	resample.SetInterpolator = interpolator
	resample.SetOutputDirection(ref_img.GetDirection())
	resample.SetOutputOrigin(ref_img.GetOrigin())
	new_spacing = [*ref_img.GetSpacing()[:2],new_z_spacing]
	resample.SetOutputSpacing(new_spacing)
	new_n_slices = int(np.ceil(ref_img.GetSize()[-1]*ref_img.GetSpacing()[-1]/new_z_spacing))
	resample.SetSize((*ref_img.GetSize()[:2], new_n_slices))
	ref_img = resample.Execute(ref_img)
	imgs_out = []
	for img in imgs:
		imgs_out.append(resample.Execute(img))
	return ref_img, imgs_out

def gauss_filter_downsample(img, new_spacing, device='cuda', ttype=torch.float32, max_ksize=None):
	# alternative: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/b75721a121102cf972f942fad927751089a7cc80/Python/05_Results_Visualization.ipynb
	#get 3d spacing to go to
	if isinstance(new_spacing,int) or isinstance(new_spacing,float):
		new_spacing = [new_spacing,new_spacing,new_spacing]
	
	#invert spacing for numpy array style
	NS = [new_spacing[2],new_spacing[1],new_spacing[0]] 
	x,y,z = list(img.GetSpacing())
	SP = [z,y,x]
	vox_red = [] #kernel size that can be used for slice reduction
	mid = [] #middle index of kernel
	for sp,ns in zip(SP,NS): #spacing, new spacing
		size = round(ns/sp)
		#get odd number for size
		if not is_odd(size):
			size = make_odd(size)
		size = max(size,1)
		use_size = int(size)
		if max_ksize is not None:
			use_size = min(use_size,max_ksize)

		vox_red.append(use_size)
		mid.append(int(np.floor(size/2)))
	
	#create gaussian kernel for convolution
	n = np.zeros(vox_red)
	n[mid[0],mid[1],mid[2]] = 1
	kernel = gaussian_filter(n,sigma=np.array(vox_red)/6,order=0, mode='constant',cval=0)
	kernel = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
	#normalize kernel to sum up to 1
	kernel = (kernel*(1/kernel.sum()))
	#use torch for conv since it can be done on gpu
	conv = nn.Conv3d(1, 1, kernel_size=vox_red,stride=1, padding='same', bias=False)
	conv.weight = nn.Parameter(kernel)            
	conv = conv.type(ttype).to(device)
	for param in conv.parameters():
		param.requires_grad = False
	
	#use input_img
	input_img = sitk.GetArrayFromImage(img)
	mn_hu = input_img.min()
	if mn_hu<0: #add min (conv can not handle very negative values properly)
		input_img += mn_hu*-1
	input_img = torch.Tensor(input_img).unsqueeze(0).unsqueeze(0).type(ttype).to(device)
	new_img = rtrn_np(conv(input_img))[0,0]
	if mn_hu<0: #subtract min again
		new_img += mn_hu
	new_img = np2sitk(new_img,img)
	return Resample_img(new_img, new_spacing=new_spacing, interpolator=sitk.sitkBSpline)

def tolerance_adj(img, digits=4):
	"""
	Adjusts tolerance of floats in image
	"""
	spacing = [round(sp, digits) for sp in img.GetSpacing()]
	direction = [round(d, digits) for d in img.GetDirection()]
	origin = [round(o, digits) for o in img.GetOrigin()]

	img.SetSpacing(spacing)
	img.SetDirection(direction)
	img.SetOrigin(origin)
	return img





