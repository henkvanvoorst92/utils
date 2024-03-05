import numpy as np
import os
import SimpleITK as sitk
import itk

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


def dcm2niix(filename, input_dir, output_dir, dcm2niix=None, add_args=''):
	#
    # if no path defined to dcm2niix executable
    if dcm2niix is None:
        dcm2niix = 'dcm2niix'

    command = dcm2niix + " -f " + filename + " -i y -z y -o" + ' "' + output_dir + '" "' + input_dir + '"' + add_args
    os.system(command)

def find_volume_nii(p, remove_small=None):
    #if remove is an int small sized images are removed
    maxz = 0
    for f in os.listdir(p):
        if not '.nii' in f:
            continue
        f = os.path.join(p,f)
        img = sitk.ReadImage(f)
        #print(f,img.GetSize())
        if img.GetSize()[-1]>maxz:
            maxz = img.GetSize()[-1]
            img_out = img
            file_out = f
        if remove_small is not None:
            if img.GetSize()[-1]<remove_small:
                os.remove(f)
    return file_out,img_out


def assert_resliced_or_tilted(path,scanname='NCCT', ID='',file=None):
	resl_tilted = [os.path.join(path,f) for f in os.listdir(path) \
				   if ('tilt' in f.lower() or 'eq' in f.lower()) and scanname.lower() in f.lower()]
	if len(resl_tilted)>0:
		p_ncct = resl_tilted [0]
		print(ID, scanname,'tilted or resliced:', '\n', p_ncct, '\n n adjusted:',len(resl_tilted))
		adjusted = True
	else:
		if file is None:
			p_ncct = os.path.join(path,scanname+'.nii.gz')
		else:
			p_ncct = file
		adjusted = False
	return p_ncct, adjusted

def sitk_flip_AP(img: sitk.SimpleITK.Image):
	return sitk.Flip(img, [False, True, False])

def sitk_rotate(image, degrees, interpolator=sitk.sitkLinear):

    size = image.GetSize()
    spacing = image.GetSpacing()
    center = (size[0] * spacing[0] / 2.0, size[1] * spacing[1] / 2.0, size[2] * spacing[2] / 2.0)

    # Create a 3D Euler transformation
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(0, 0, np.radians(degrees))  # Rotation around the Z-axis by 30 degrees

    # Resample the image using the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampled_image = resampler.Execute(image)

    return resampled_image

def sitk_flip_lateral(img: sitk.SimpleITK.Image):
    return sitk.Flip(img, [True,False,False])

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

#normalization of images
def np_znorm(arr):
    return (arr-arr.mean())/arr.std()

def sitk_znorm(img):
    return np2sitk(np_znorm(sitk.GetArrayFromImage(img)),img)


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


def resample_registered_images(template_image, adjust_image, interpolator=sitk.sitkNearestNeighbor):
	"""
    resamples and crops the adjust_image to the template image

    use interpolator NearestNeighbor for masks, Bspline3 for images as adjust_image
    """
	# Initialize the ResampleImageFilter
	resample = sitk.ResampleImageFilter()
	# Set the template image (formerly known as mra_image) as the reference
	resample.SetReferenceImage(template_image)

	# Set the interpolation method to linear
	resample.SetInterpolator(interpolator)

	# Use an affine transform for resampling the adjust image (formerly known as t1_image)
	resample.SetTransform(sitk.AffineTransform(adjust_image.GetDimension()))

	# Match the spacing, size, direction, and origin to the template image
	resample.SetOutputSpacing(template_image.GetSpacing())
	resample.SetSize(template_image.GetSize())
	resample.SetOutputDirection(template_image.GetDirection())
	resample.SetOutputOrigin(template_image.GetOrigin())

	# Execute the resampling on the adjust image
	resampled_adjust_image = resample.Execute(adjust_image)

	return resampled_adjust_image


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

def sitk2itk(sitk_image):
	sitk_array = sitk.GetArrayFromImage(sitk_image)
	itk_image = itk.GetImageFromArray(sitk_array)
	# SimpleITK and ITK have different conventions for image axes,
	# so we may need to permute the axes of the ITK image to match the SimpleITK image
	itk_image.SetSpacing(sitk_image.GetSpacing())
	itk_image.SetOrigin(sitk_image.GetOrigin())

	# Convert SimpleITK direction to ITK direction
	direction = itk.matrix_from_array(np.array(sitk_image.GetDirection()).reshape(sitk_image.GetDimension(), -1))
	#itk_image.SetDirection(direction)
	return itk_image

def itk2sitk(itk_image):
	# ITK stores image data as a 1D array, so we need to reshape it to the correct dimensions
	itk_array = itk.array_from_image(itk_image)
	# Convert the numpy array to a SimpleITK Image
	sitk_image = sitk.GetImageFromArray(itk_array)
	# Set the spacing, origin, and direction to match the ITK image
	sitk_image.SetSpacing(list(itk_image.GetSpacing()))
	sitk_image.SetOrigin(list(itk_image.GetOrigin()))

	# ITK direction cosines need to be converted to a SimpleITK direction
	# direction = np.array(itk_image.GetDirection()).flatten()
	# if itk_image.GetImageDimension() == 3:
	#     # For 3D image, direction cosines need to be reordered
	#     direction = direction[[0,3,6,1,4,7,2,5,8]]
	# sitk_image.SetDirection(tuple(direction))

	return sitk_image

def sitk_add_tags(img, mdata, tags=['ExposureinmAs','KVP','CTDIvol']):
    #add tags from pd.Dataframe (mdata) to image
    for tag in tags:
        tag_value = str(list(mdata['KVP'].values))
        img.SetMetaData(tag,tag_value)
    return img


def n4_bias_field_correction(img, number_of_iterations=None):
    # Read the image
    img = sitk.Cast(img, sitk.sitkFloat32)

    # Apply N4 Bias Field Correction
    # The mask image is optional, if not provided the algorithm
    # will compute a mask internally.
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # Number of iterations per resolution level, more iterations
    # may yield better results but will take longer to compute.
    if number_of_iterations is None:
        number_of_iterations = [50, 50, 50, 50]
    corrector.SetMaximumNumberOfIterations(number_of_iterations)

    return corrector.Execute(img)
def otsu_threshold(image, return_mask=False):
	# Apply Otsu threshold
	otsu_filter = sitk.OtsuThresholdImageFilter()
	otsu_filter.SetInsideValue(0)
	otsu_filter.SetOutsideValue(1)
	otsu_threshold_mask = otsu_filter.Execute(image)

	otsu_threshold_value = otsu_filter.GetThreshold()

	if return_mask:
		out = otsu_threshold_mask, otsu_threshold_value
	else:
		out = otsu_threshold_value

	return out

def ctp_exposure_weights(exposures):
    exposures = np.array(exposures)
    tot = exposures.sum()
    weights = exposures / tot
    return weights


# t = mdata['AcquisitionDateTime']
# td = mdata['timedifference']
# exposures = mdata['ExposureinmAs'].astype(np.float32).values

# wt_middle = 1/3

# for i in range(1,len(exposures)-1):
#     exps = exposures[i-1:i+2]
#     expw = exps/exps.sum()
#     if t is not None:
#         td1, td3 = td[i], td[i+1]
#         tot = td1+td3
#         w1 = (1-td1/tot)*(2/3)
#         w3 = (1-td3/tot)*(2/3)
#         tw = np.array([w1,1/3,w3])
#     else:
#         tw = [1/3,1/3,1/3]
#     w = tw*expw
#     w = w/w.sum()
#     break
# print(tw,expw,w)
# td