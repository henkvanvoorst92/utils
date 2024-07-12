import numpy as np
import itertools
import sys
import os
import torch
from scipy.ndimage import binary_dilation
from torch import nn
import SimpleITK as sitk
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import gaussian_filter
sys.path.append('..')
from utils.distances import get_3Dmask_coordinates, nearest_neighbor_distances, coordinates2mask_3D
from utils.utils import np2sitk

def sitk_dilate_mask(mask,radius_mm, dilate_2D=False):
	
	radius_3d = [int(np.floor(radius_mm / mask.GetSpacing()[0])),
			 int(np.floor(radius_mm / mask.GetSpacing()[1])),
			 int(np.floor(radius_mm / mask.GetSpacing()[2]))]
	if dilate_2D:
		radius_3d[2] = 0
	
	dilate = sitk.BinaryDilateImageFilter()
	dilate.SetBackgroundValue(0)
	dilate.SetForegroundValue(1)
	dilate.SetKernelRadius(radius_3d)
	return dilate.Execute(mask)

def sitk_dilate_mm(mask,kernel_mm, background=0, foreground=1):
	
	if isinstance(kernel_mm,int) or isinstance(kernel_mm,float):
		k0 = k1 = k2 = kernel_mm
	elif isinstance(kernel_mm,tuple) or isinstance(kernel_mm,list):
		k0, k1, k2 = kernel_mm
	
	kernel_rad = (int(np.floor(k0/mask.GetSpacing()[0])),
			 int(np.floor(k1/mask.GetSpacing()[1])),
			 int(np.floor(k2/mask.GetSpacing()[2])))
	
	dilate = sitk.BinaryDilateImageFilter()
	dilate.SetBackgroundValue(background)
	dilate.SetForegroundValue(foreground)
	dilate.SetKernelRadius(kernel_rad)
	return dilate.Execute(mask)

def sitk_erode_mm(mask,kernel_mm, background=0, foreground=1):
	
	if isinstance(kernel_mm,int) or isinstance(kernel_mm,float):
		k0 = k1 = k2 = kernel_mm
	elif isinstance(kernel_mm,tuple) or isinstance(kernel_mm,list):
		k0, k1, k2 = kernel_mm
	
	kernel_rad = (int(np.floor(k0/mask.GetSpacing()[0])),
			 int(np.floor(k1/mask.GetSpacing()[1])),
			 int(np.floor(k2/mask.GetSpacing()[2])))
	
	erode = sitk.BinaryErodeImageFilter()
	erode.SetBackgroundValue(background)
	erode.SetForegroundValue(foreground)
	erode.SetKernelRadius(kernel_rad)
	return erode.Execute(mask)

def compute_volume(mask: sitk.SimpleITK.Image):
	# mask is an sitk image
	# used to compute the volume in ml for foreground
	sp = mask.GetSpacing()
	vol_per_vox = sp[0]*sp[1]*sp[2]
	
	m = sitk.GetArrayFromImage(mask)
	voxels = m.sum()
	#volume in ml
	tot_volume = vol_per_vox*voxels/1000
	return tot_volume

def np_largest_cc(seg):
	labels, nc = label(seg) # uses scipy.ndimage.measurements
	unique, counts = np.unique(labels, return_counts=True) # unique connected components
	unique, counts = unique[1:], counts[1:]
	v = unique[np.argmax(counts)] # cc's larger than min_count
	return (labels==v)*1

def np_n_largest_cc(mask,n_top):
    labels, nc = label(mask) # uses scipy.ndimage.measurements
    unique, counts = np.unique(labels, return_counts=True) # unique connected components
    unique, counts = unique[1:], counts[1:]
    top_ixs = np.argpartition(counts, n_top*-1)[n_top*-1:]+1
    return np.isin(labels,top_ixs)*1


def sitk_select_components_minsize(mask: sitk.SimpleITK.Image,  # sitk mask to extract connected components from
								   min_vol_ml=1):
	# minimum number of voxel in the array
	min_count = np.ceil(min_vol_ml / np.product(np.array(mask.GetSpacing()))).astype(int)

	# compute connected components and sort for faster processing
	component_image = sitk.ConnectedComponent(mask)
	SC = sitk.RelabelComponent(component_image, sortByObjectSize=True)
	sc = sitk.GetArrayFromImage(SC)
	ccs2use = []
	for cc in np.unique(sc)[1:]:
		count = sitk.GetArrayFromImage(SC == cc).sum()
		if count < min_count:
			break
		else:
			ccs2use.append(cc)

	lcc = (SC > 0) & (SC <= ccs2use[-1])
	return lcc

def remove_small_cc(seg,min_count=100):
	# filters out small connected components
	labels, nc = label(seg) # uses scipy.ndimage.measurements
	unique, counts = np.unique(labels, return_counts=True) # unique connected components
	v = unique[counts>min_count] # cc's larger than min_count
	out = ((labels*np.isin(labels,v)*1)*(seg>0)>0)*1 #combine all the above in a binary segmentation
	return out

def get_min_voxcount(min_cc_vol,spacing):
	vol_per_vox = np.prod(np.array(spacing))
	return [int(round(cc/vol_per_vox)) for cc in min_cc_vol]

def get_largest_cc(mask):
	labels, nc = label(mask) # uses scipy.ndimage.measurements
	unique, counts = np.unique(labels, return_counts=True) # unique connected components
	unique, counts = unique[1:], counts[1:]
	ix = np.argmax(counts)+1
	out = (labels==ix)*1
	return out

def lower_upper_ix(mask,
				   z_difference=150,
				   z = -1,
				   foreground=1,
				   min_area=1000):

	"""
	Runs in a max from top to bottom of the head, finds
	the top slice where area>min_area and goes down 
	subsequently to z_difference mm below that point to
	return the bottom slice
	"""
	
	zdim = mask.GetSize()[z]
	for i in range(0,zdim):
		slice_id = zdim - i - 1
		slice_mask = mask[:, :, slice_id]
		label_info_filter = sitk.LabelStatisticsImageFilter()
		label_info_filter.Execute(slice_mask,slice_mask)
		area = label_info_filter.GetCount(foreground) * mask.GetSpacing()[0] * mask.GetSpacing()[1]
		if area>min_area:
			break
	top_slice = slice_id
	max_distance = mask.GetSpacing()[z]*zdim # the z-distance that is available in the mask
	bottom_slice = top_slice - int(z_difference/mask.GetSpacing()[z])
	if bottom_slice<0:
		bottom_slice = 0
			
	return (bottom_slice, top_slice, area, max_distance)


def np_slicewise(mask, funcs, repeats=1, dim=0):
	"""
	Applies a list of functions iteratively (repeats) slice by slice of an 3D np volume
	mask: mask to do operation on
	funcs: list of functions applied consecutively
	repeats: each function is applied for a set number
	dim: dimension to do operation over (default is z dim)
	"""
	if isinstance(mask,sitk.SimpleITK.Image):
		mask = sitk.GetArrayFromImage(mask)

	out = np.zeros_like(mask)
	for sliceno in range(mask.shape[dim]):
		if dim==0:
			m = mask[sliceno,:,:]
		elif dim==1:
			m = mask[:,sliceno,:]
		elif dim==2:
			m = mask[:,:,sliceno]
		for r in range(repeats):
			for func in funcs:
				m = func(m)
		if dim==0:
			out[sliceno,:,:] = m
		elif dim==1:
			out[:,sliceno,:] = m
		elif dim==2:
			out[:,:,sliceno] = m
	return out

def np_multidim_slicewise(mask,funcs,repeats=1,dims=[0,1,2,0,1,2]):
	#runs np slicewise consecutively over dims
	#dims: a list with dimensions running each dim once
	for d in dims:
		mask = np_slicewise(mask, funcs, repeats=repeats, dim=d)
	return mask

def get_list_boundaries(dct, ixs=[0,1,2]):
	out = []
	for i in ixs:
		out.extend([dct[i]['lower'],dct[i]['upper']])
	return out


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
		tmp = (int(np.floor(rad / mask.GetSpacing()[0])),
			   int(np.floor(rad / mask.GetSpacing()[1])),
			   int(np.floor(rad / mask.GetSpacing()[2])))
		rads_3d.append(tmp)

	for r, oper in zip(rads_3d, operations):
		oper.SetBackgroundValue(abs(foreground - 1))
		oper.SetForegroundValue(foreground)
		oper.SetKernelRadius(r)
		mask = oper.Execute(mask)

	return mask


## this code comes from the CTA2NCCT train file and might interfere with similar named functions above
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
		if iters>=0:
			self.operation = 'dilate'
			self.iters = iters
		else:
			self.operation = 'erode'
			self.iters = abs(iters)

		self.background = background
		self.type = type
		self.device = device
		self.connectivity = connect
		self.combine = combine # how to combine multiple images into a mask
		
		if connect==8:
			if not self.dim3D:
				self.kernel = torch.tensor([
						[1, 1, 1],
						[1, 1, 1],
						[1, 1, 1] ], 
						dtype=self.type
						).unsqueeze(0).unsqueeze(0) # shape: (1, 1, 3, 3) = (Batch,Channel,H,W)
			else:
				self.kernel = torch.ones([3,3,3],dtype=self.type).unsqueeze(0).unsqueeze(0) # shape: (1, 1, 3, 3, 3) = (Batch,Channel,D,H,W)
			
		elif connect==4:
			if not self.dim3D:
				self.kernel = torch.tensor([
						[0, 1, 0],
						[1, 1, 1],
						[0, 1, 0] ], 
						dtype=self.type
						).unsqueeze(0).unsqueeze(0) # shape: (1, 1, 3, 3) = (Batch,Channel,H,W)
			else:
				self.kernel = torch.tensor([[[0, 1, 0],[1, 1, 1],[0, 1, 0]],
									[[0, 1, 0],[1, 1, 1],[0, 1, 0]],
									[[0, 1, 0],[1, 1, 1],[0, 1, 0]]],dtype=self.type
									).unsqueeze(0).unsqueeze(0) # shape: (1, 1, 3, 3,3) = (Batch,Channel,D,H,W)
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
	def __call__(self,img):
		# first construct the mask (foreground=1) from the image
		mask = (img>self.background).type(self.type)
		if self.iters>0:
			# repeatedly erode or dilate
			for i in range(self.iters):
				if self.operation=='dilate':
					mask = (self.conv(mask)>0).type(self.type)
				if self.operation=='erode':
					mask = (self.conv(mask)>=self.connectivity).type(self.type)
		# final threshold after erosion or dilation
		return mask

def boundary_mask(mask, dims=None, foregroundval=1):
	"""
	Function to identify for each dimension of a 3D
	mask what the lower and upper starting points
	are for the foreground. This can be used to crop
	the image for storage and loading purposes.

	Output:
	dictionary with: dct[dimension] = {lower:value, upper:value}

	"""

	if dims is None:
		dims = [0, 1, 2]

	dimdct = {}
	for dim in dims:
		tmpdct = {}
		notfoundm1, notfoundm2 = True, True
		s = int(mask.shape[dim])
		for i in range(1, int(s)):
			if dim == 0:
				m1 = mask[i, :, :].max()
				m2 = mask[s - i, :, :].max()
			elif dim == 1:
				m1 = mask[:, i, :].max()
				m2 = mask[:, s - i, :].max()
			elif dim == 2:
				m1 = mask[:, :, i].max()
				m2 = mask[:, :, s - i].max()

			if (m1 >= foregroundval) & notfoundm1:
				tmpdct['lower'] = i
				notfoundm1 = False
			if (m2 >= foregroundval) & notfoundm2:
				tmpdct['upper'] = s - i
				notfoundm2 = False
			if (notfoundm1 == False) & (notfoundm2 == False):
				break
		dimdct[dim] = tmpdct
	return dimdct

def adjust_margin_boundary_mask_dct(dct,
									margin=np.array([10, 10, 10]),
									img=None):
	# adds a certain margin to the boundary mask dct
	# if img is an sitk.Image the values in margin are in mm
	# otherwise the values in margin are pixels to add/subtract to boundaries
	if isinstance(img, sitk.Image) and not img is None:
		# retrieve spacing and adjust margin
		sp1, sp2, sp0 = img.GetSpacing()
		sp = np.array([sp0, sp1, sp2])
		margin = np.ceil(margin / sp).astype(int)

	for k in dct.keys():
		if isinstance(k, int):
			if k < 3:
				dct[k] = {'lower': max(dct[k]['lower'] - margin[k], 0),
						  'upper': dct[k]['upper'] + margin[k]}
	return dct

def batch_boundaries(batch):
	"""
	Computes per batch of masks the boundaries
	"""
	if isinstance(batch, torch.Tensor):
		batch = batch.numpy()
		
	boundaries = []
	for i in range(batch.shape[0]):
		mask = batch[i,:,:]
		dct = boundary_mask(np.expand_dims(mask,2),dims=[0,1])
		b = [[dct[0]['lower'],dct[0]['upper']],
			 [dct[1]['lower'],dct[1]['upper']]]
		boundaries.append(b)
	
	boundaries = np.stack(boundaries).astype(batch.dtype)
	return boundaries

def local_mask(mask,margin):
	# computes a square mask surrounding the foreground
	# adds a margin around min and max indices of mask
	bd = boundary_mask(mask,dims=[0,1,2],foregroundval=1)
	m2 = np.zeros_like(mask)
	m2[bd[0]['lower']-margin:bd[0]['upper']+margin,
	   bd[1]['lower']-margin:bd[1]['upper']+margin,
	   bd[2]['lower']-margin:bd[2]['upper']+margin] = 1
	return m2

def staple(segmentations, foregroundValue = 1, threshold = 0.5):
	#https://colab.research.google.com/github/matjesg/deepflash2/blob/master/nbs/09_gt.ipynb#scrollTo=66AXN5S-fbmQ
	#https://towardsdatascience.com/how-to-use-the-staple-algorithm-to-combine-multiple-image-segmentations-ce91ebeb451e
	'STAPLE: Simultaneous Truth and Performance Level Estimation with simple ITK'
	#sitk = import_sitk()
	if np.all([isinstance(s,np.ndarray) for s in segmentations]):
		segmentations = [sitk.GetImageFromArray(x) for x in segmentations]
	STAPLE_probabilities = sitk.STAPLE(segmentations)
	STAPLE = STAPLE_probabilities > threshold
	#STAPLE = sitk.GetArrayViewFromImage(STAPLE)
	return sitk.GetArrayFromImage(STAPLE)


def np_volume(mask:np.ndarray,spacing):
	vol_per_vox = np.prod(np.array(spacing))
	return vol_per_vox*mask.sum()

def get_nonzero_slices(mask:np.ndarray):
	#returns indices of slices that are not zero
	mx = mask.argmax(axis=1).argmax(axis=1)
	return [i for i in range(len(mx)) if mx[i]>0]

def torch_init_gauss_conv3d(kernel_size,crop_bound=0,device='cpu',ttype=torch.float32, pad='same'):
	#kernel_size (int/tuple): size of 3 dims of a cube
	#crop_bound: if>0 the boundaries of all dims are set to 0 (excludes edges artefacts when used for inference)
	#returns conv layer with gauss weights and the kernel
	if isinstance(kernel_size,int):
		kernel_size = [kernel_size,kernel_size,kernel_size]
	
	low = int(kernel_size[0]/2)-1
	high = low+1

	mc1 = [coords for coords in itertools.combinations_with_replacement([low,high,low],3)]
	mc2 = [coords for coords in itertools.combinations_with_replacement([high,low,high],3)]
	mid_coords = list(set([*mc1,*mc2]))
	n = np.zeros(kernel_size)
	for mc in mid_coords:
		 n[mc] = 1
	kernel = gaussian_filter(n,sigma=np.array(kernel_size)/6,order=0, mode='constant',cval=0)
	if crop_bound>0:
		# set weights of boundaries to zero to exclude edges
		m = np.zeros(kernel_size)
		m[crop_bound:-crop_bound,crop_bound:-crop_bound,crop_bound:-crop_bound] = 1
		kernel = kernel*m
	#acertain weighting adds up to one
	kernel = (kernel*(1/kernel.sum()))
	kernel = torch.tensor(kernel).unsqueeze(0).unsqueeze(0).type(ttype).to(device)
	
	conv = nn.Conv3d(1, 1, kernel_size=kernel_size,stride=1, padding=pad, bias=False)
	conv.weight = nn.Parameter(kernel)            
	
	conv = conv.type(ttype).to(device)
	for param in conv.parameters():
		param.requires_grad = False
	return conv,kernel

def lesion_count_volstat(seg,spacing=None,min_size=1):
	#counts the number of separate lesions in segmentation (seg)
	#and computs avg and median volume per lesion
	#if spacing is geven min_size is in mm3
	if spacing is not None:
		zsp,ysp,xsp = spacing
		vol_per_vox = zsp*ysp*xsp
		min_size = min_size/vol_per_vox
	# filters out small connected components
	labels, nc = label(seg) # uses scipy.ndimage.measurements
	unique, counts = np.unique(labels, return_counts=True) # unique connected components
	v = unique[counts>=min_size] # cc's larger than min_count
	counts = counts[(counts>min_size)&(unique!=0)]
	
	n_lesions = len(counts)
	avg_size = np.mean(counts)
	median_size = np.median(counts)
	if spacing is not None:
		avg_size = avg_size*vol_per_vox
		median_size = median_size*vol_per_vox
		
	return n_lesions, avg_size, median_size

def tolerance_adj(img,digits=4):
	spacing = [round(sp,digits) for sp in img.GetSpacing()]
	direction = [round(d,digits) for d in img.GetDirection()]
	origin = [round(o,digits) for o in img.GetOrigin()]
	
	img.SetSpacing(spacing)
	img.SetDirection(direction)
	img.SetOrigin(origin)
	return img
def mask2coordinates(mask):
	return np.vstack(np.where(mask)).T


def distance_to_skull_map(mask: np.ndarray,
						  brainmask: np.ndarray,
						  spacing: np.ndarray or tuple or list = None):
	"""
    Computes the distance of all voxels in mask to the skull
    mask : can be vessel segmentation or again the brainmask
            --> used to compute distance --> smaller foreground = faster compute
    brainmask : edge of the brainmask is used to represent the skull
    """

	mask_coords = get_3Dmask_coordinates(mask)

	edge = binary_dilation(brainmask, structure=np.ones((3, 3, 3)), iterations=1) - brainmask
	edge_coords = get_3Dmask_coordinates(edge)

	dists_per_coord = nearest_neighbor_distances(
		np.array(mask_coords).astype(np.int16),
		np.array(edge_coords).astype(np.int16),
		spacing=spacing
	)
	distance_map = coordinates2mask_3D(np.array(mask_coords),
									   np.zeros_like(mask),
									   dists_per_coord)

	return distance_map

def nrrd_segment_numbers(nrrd_image):
    out = []
    for k in nrrd_image.GetMetaDataKeys():
        k = (k.split('_')[0]).replace('Segment','')
        try:
            out.append(int(k))
        except:
            continue
    return list(set(out))

def nrrd_label_dict(seg_nrrd_file,only_renamed_segments=True):
    nrrd_image = sitk.ReadImage(seg_nrrd_file)

    # Get all metadata keys
    metadata_keys = nrrd_image.GetMetaDataKeys()

    segnos = nrrd_segment_numbers(nrrd_image)
    # Initialize a dictionary to hold segment names and label values
    segment_dict = {}
    for segment_index in segnos:
        segment_name = nrrd_image.GetMetaData(f'Segment{segment_index}_Name')
        #do not considered non-renamed segments
        if only_renamed_segments:
            if 'Segment' in segment_name:
                continue
        segment_label_value = int(nrrd_image.GetMetaData(f'Segment{segment_index}_LabelValue'))
        segment_dict[segment_name] = segment_label_value
    return segment_dict


def multimask2singlemask(mask, value, p_sav=None):
	"""
    Selects a subclass (value) of the a mask with multiple values (mask)
    saves or returns it
    """
	if isinstance(mask, str):
		file = os.path.basename(mask)
		if p_sav is not None:
			os.makedirs(p_sav, exist_ok=True)
			p_sav = os.path.join(p_sav, file)
		mask = sitk.ReadImage(mask)
	mask = sitk.Cast((mask == value) * 1, sitk.sitkInt16)

	sitk.WriteImage(mask, p_sav)
	return mask

def select_lesion_in_roimask(roimask, lesion):
	"""
    Separates lesion in connected components
    returns a new mask of lesion components that are
    """
	if isinstance(roimask, sitk.Image):
		roimask = sitk.GetArrayFromImage(roimask)
	if isinstance(lesion, sitk.Image):
		lesion = sitk.GetArrayFromImage(lesion)

	labels, nc = label(lesion)
	labels_in_roimask = np.unique(labels * roimask)
	return np.isin(labels, labels_in_roimask) * 1

def lesion_in_roimask_value(roimask, lesion):
    """
    Given a lesion and roimask, identifies the roimask subclass
    with the largest portion of the lesion
    returns values/counts dict for each separate connected component ordered by count
    """
    if isinstance(roimask, sitk.Image):
        roimask = sitk.GetArrayFromImage(roimask)
    if isinstance(lesion, sitk.Image):
        lesion = sitk.GetArrayFromImage(lesion)

    vals, counts = np.unique(roimask * lesion, return_counts=True)
    ixs = np.argsort(counts)[::-1]
    vc_dct = {vals[ix]:counts[ix] for ix in ixs if vals[ix]!=0}

    return vc_dct

def lesion_in_infarcted_hemisphere(hemispheres,
                                   identification_mask,
                                   lesion, dil_mm=5, max_contralater_lesion_ml=10):
	"""
	hemispheres is an sitk.Image mask representing left and right hemispheres
	identification_mask is an sitk.Image mask for selecting the right hemisphere
	lesion is an sitk.Image mask with separate distinct connected components of the lesion
	returns only separate parts of the lesion in the infarcted hemisphere
	"""
	# select the infarcted half of the brain
	count_dct = lesion_in_roimask_value(hemispheres, identification_mask)
	if len(count_dct)==0:
		infarcted_hemisphere = sitk_dilate_mm((hemispheres>0) * 1, dil_mm)
	else:
		select_roival = list(count_dct.keys())[0]
		infarcted_hemisphere = sitk_dilate_mm((hemispheres == select_roival) * 1, dil_mm)

	if len(count_dct)>1:
		contralateral_lesion_volume_ml = list(count_dct.values())[1]*np.prod(hemispheres.GetSpacing())/1000
		#sanity check to see of not large lesion in contralateral hemisphere
		if contralateral_lesion_volume_ml>max_contralater_lesion_ml:
			print('contralateral lesion presens, volume ml:', contralateral_lesion_volume_ml)
	# select lesions with part in the infarcted hemisphere
	lesion_in_hemisphere = select_lesion_in_roimask(infarcted_hemisphere, lesion)
	lesion_in_hemisphere = sitk.Cast(np2sitk(lesion_in_hemisphere, hemispheres), sitk.sitkInt16)
	return lesion_in_hemisphere, infarcted_hemisphere
