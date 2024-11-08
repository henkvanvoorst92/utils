
import SimpleITK as sitk
import itk #should be version with itk-elastix
import sys
import argparse
import os
import ants
sys.path.append('..')
from utils.preprocess import sitk2itk, itk2sitk
from utils.ants_utils import sitk2ants, ants2sitk


def ants_register(fixed,
                  moving,
                  addname='',
                  rp=None,
                  pid=None,
                  p_transform=None,
                  clip_range=None,
                  return_tx=False):
    """
    Registers the atlas to the scan
    then uses the transformation matrix
    to register the cow mask to the scan

    scan: an mra or cta scan
    atlas: mra vessel atlas allligned with cow mask
    cow: Circle of willis mask to select arteries
    addname: name to add to the cow save file
    rp: registration parameter settings

    returns and/or saves the registered cow
    """

    clipped_register = False
    if clip_range is not None:
        if isinstance(fixed, sitk.Image) and isinstance(moving, sitk.Image):
            fixed_reg = sitk.Image(sitk.Clamp(fixed, lowerBound=clip_range[0], upperBound=clip_range[1]))
            fixed_reg = sitk2ants(sitk.Cast(fixed_reg, sitk.sitkFloat32))
            moving_reg = sitk.Image(sitk.Clamp(moving, lowerBound=clip_range[0], upperBound=clip_range[1]))
            moving_reg = sitk2ants(sitk.Cast(moving_reg, sitk.sitkFloat32))
            clipped_register = True

    if isinstance(fixed, sitk.Image):
        fixed = sitk2ants(sitk.Cast(fixed, sitk.sitkFloat32))

    if isinstance(moving, sitk.Image):
        moving = sitk2ants(sitk.Cast(sitk.Image(moving), sitk.sitkFloat32))

    if rp is None:
        print('Using default rp')
        rp = {'type_of_transform': 'TRSAA',
              'fix_bm': None,
              'mv_bm': None,
              'metric': 'mattes',
              'mask_all_stages': False,
              'default_value': 0,
              'interpolator': 'linear' #or 'nearestNeighbor''bSpline'
              }
        print(rp)
        print('Using default estimator')

    if clipped_register:
        mytx = ants.registration(fixed_reg, moving_reg,
                                 type_of_transform=rp['type_of_transform'],
                                 mask=rp['fix_bm'],
                                 moving_mask=rp['mv_bm'],
                                 mask_all_stages=rp['mask_all_stages'],
                                 aff_metric=rp['metric'],
                                 syn_metric=rp['metric']
                                 )

    else:
        mytx = ants.registration(fixed, moving,
                                 type_of_transform=rp['type_of_transform'],
                                 mask=rp['fix_bm'],
                                 moving_mask=rp['mv_bm'],
                                 mask_all_stages=rp['mask_all_stages'],
                                 aff_metric=rp['metric'],
                                 syn_metric=rp['metric']
                                 )

    # apply the transform to the cow
    moved = ants.apply_transforms(fixed, moving,
                                  interpolator=rp['interpolator'],
                                  transformlist=mytx['fwdtransforms'],
                                  default_value=rp['default_value']
                                  )

    if pid is not None:
        pid_reg = os.path.join(pid, 'registration')
        if not os.path.exists(pid_reg):
            os.makedirs(pid_reg)
        ants.image_write(moved, os.path.join(pid, '{}moved.nii.gz'.format(addname)))
    if p_transform is not None:
        if 'SyN' in rp['type_of_transform']:
            ix = 1
        else:
            ix = 0
        ants.write_transform(ants.read_transform(mytx['fwdtransforms'][ix]),
                             os.path.join(p_transform , '{}fwd.mat'.format(addname)))
        ants.write_transform(ants.read_transform(mytx['invtransforms'][0]),
                             os.path.join(p_transform, '{}inv.mat'.format(addname)))
    if return_tx:
        return sitk.Cast(ants2sitk(moved), sitk.sitkInt16), mytx
    else:
        return sitk.Cast(ants2sitk(moved), sitk.sitkInt16)

def ants_register_roi_atlas(fixed,
                            moving,  # the atlas image
                            roi,  # the roi mask in the atlas
                            addname='',
                            rp=None,
                            pid=None,
                            p_transform=None,
                            clip_range=None):
    """
    Registers the atlas (moving) to the scan (fixed)
    then uses the transformation matrix
    to register the cow mask to the scan

    scan: an mra or cta scan
    atlas: mra vessel atlas allligned with cow mask
    cow: Circle of willis mask to select arteries
    addname: name to add to the cow save file
    rp: registration parameter settings

    returns and/or saves the registered cow
    """

    if clip_range is not None:
        if isinstance(fixed, sitk.Image) and isinstance(moving, sitk.Image):
            fixed = sitk.Image(sitk.Clamp(sitk.Image(fixed),
                                          lowerBound=clip_range[0], upperBound=clip_range[1]))
            moving = sitk.Image(sitk.Clamp(sitk.Image(moving),
                                           lowerBound=clip_range[0], upperBound=clip_range[1]))

    if isinstance(fixed, sitk.Image):
        fixed = sitk2ants(sitk.Cast(sitk.Image(fixed), sitk.sitkFloat32))

    if isinstance(moving, sitk.Image):
        moving = sitk2ants(sitk.Cast(sitk.Image(moving), sitk.sitkFloat32))

    if isinstance(roi, sitk.Image):
        roi = sitk2ants(sitk.Cast(sitk.Image(roi), sitk.sitkFloat32))

    if rp is None:
        print('Using default rp')
        rp = {'type_of_transform': 'TRSAA',
              'fix_bm': None,
              'mv_bm': None,
              'metric': 'mattes',
              'mask_all_stages': False,
              'default_value': 0,
              'interpolator': 'linear'  # or 'nearestNeighbor''bSpline'
              }
        print(rp)
        print('Using default estimator')

    mytx = ants.registration(fixed, moving,
                             type_of_transform=rp['type_of_transform'],
                             mask=rp['fix_bm'],
                             moving_mask=rp['mv_bm'],
                             mask_all_stages=rp['mask_all_stages'],
                             aff_metric=rp['metric'],
                             syn_metric=rp['metric']
                             )

    # apply the transform to the cow
    mv_roi = ants.apply_transforms(fixed,
                                   roi,
                                   interpolator='nearestNeighbor',
                                   transformlist=mytx['fwdtransforms'])

    if pid is not None:
        pid_reg = os.path.join(pid, 'registration')
        if not os.path.exists(pid_reg):
            os.makedirs(pid_reg)

        ants.image_write(mv_roi, os.path.join(pid_reg, '{}_reg.nii.gz'.format(addname)))

        ix = 1 if 'SyN' in rp['type_of_transform'] else 0
        ants.write_transform(ants.read_transform(mytx['fwdtransforms'][ix]),
                             os.path.join(p_transform , '{}fwd.mat'.format(addname)))
        ants.write_transform(ants.read_transform(mytx['invtransforms'][0]),
                             os.path.join(pid_reg, '{}inv.mat'.format(addname)))

    return sitk.Cast(ants2sitk(mv_roi), sitk.sitkInt16)

def register_roi_atlas(scan,
                       atlas,
                       roi,
                       addname='',
                       rp=None,
                       pid=None):
    """
    Registers the atlas to the scan
    then uses the transformation matrix
    to register the cow mask to the scan

    scan: an mra or cta scan
    atlas: mra vessel atlas allligned with cow mask
    roi: Circle of willis or roi mask to select arteries
    addname: name to add to the cow save file
    rp: registration parameter settings

    returns and/or saves the registered cow
    """

    if isinstance(scan, sitk.Image):
        scan = sitk2ants(sitk.Cast(scan, sitk.sitkFloat32))

    if isinstance(atlas, sitk.Image):
        atlas = sitk2ants(sitk.Cast(sitk.Image(atlas), sitk.sitkFloat32))

    if isinstance(roi, sitk.Image):
        roi = sitk2ants(sitk.Cast(sitk.Image(roi), sitk.sitkFloat32))

    mytx = ants.registration(scan, atlas,
                             type_of_transform=rp['type_of_transform'],
                             mask=rp['fix_bm'],
                             moving_mask=rp['mv_bm'],
                             mask_all_stages=rp['mask_all_stages'],
                             aff_metric=rp['metric'],
                             syn_metric=rp['metric']
                             )

    # apply the transform to the cow
    mv_roi = ants.apply_transforms(scan,
                                   roi,
                                   interpolator='nearestNeighbor',
                                   transformlist=mytx['fwdtransforms'])

    if pid is not None:
        pid_reg = os.path.join(pid, 'registration')
        if not os.path.exists(pid_reg):
            os.makedirs(pid_reg)

        ants.image_write(mv_roi, os.path.join(pid_reg, '{}_reg.nii.gz'.format(addname)))

        ix = 1 if 'SyN' in rp['type_of_transform'] else 0
        ants.write_transform(ants.read_transform(mytx['fwdtransforms'][ix]),
                             os.path.join(p_transform , '{}fwd.mat'.format(addname)))
        ants.write_transform(ants.read_transform(mytx['invtransforms'][0]),
                             os.path.join(pid_reg, '{}inv.mat'.format(addname)))

    return ants2sitk(mv_roi)


#registration with itk is 100x faster than simpleitk
#### Does not work at all!!!!!
def itk_register(fixed_image, moving_image, param_file, fix_clip=None, moving_clip=None, return_sitk=True):
    # param file is path to a .txt registration parameters file
    # this function only works for rigid and affine regs
    clip_reg = False

    # transform sitk images if required
    if isinstance(fixed_image, sitk.SimpleITK.Image):
        fixed_image = sitk2itk(fixed_image)
    if isinstance(moving_image, sitk.SimpleITK.Image):
        original_moving_image = sitk.Image(moving_image)
        moving_image = sitk2itk(moving_image)
        return_sitk = True  # if a conversion was performed an sitk image is return

    # Cast images to float for registration algorithm
    fixed_image = itk.cast_image_filter(fixed_image, ttype=(type(fixed_image), itk.Image.F3))
    moving_image = itk.cast_image_filter(moving_image, ttype=(type(moving_image), itk.Image.F3))

    # if bounds are defined registration is first performed on
    if fix_clip is not None:
        fixed_image_clip = itk.clamp_image_filter(fixed_image, bounds=clip_bounds)
        clip_reg = True
    if moving_clip is not None:
        moving_image_clip = itk.clamp_image_filter(moving_image, bounds=clip_bounds)
        clip_reg = True

    # Import Custom Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(param_file)

    # "WriteResultImage" needs to be set to "true" so that the image is resampled at the end of the registration
    # and the result_image is populated properly
    parameter_object.SetParameter(0, "WriteResultImage", "true")

    # Call registration function
    if clip_reg:
        __, transform_parameters = itk.elastix_registration_method(
            fixed_image_clip, moving_image_clip,
            parameter_object=parameter_object,
            log_to_console=False)
        result_image = itk.transformix_filter(
            moving_image,
            transform_parameters)
    else:
        result_image, transform_parameters = itk.elastix_registration_method(
            fixed_image, moving_image,
            parameter_object=parameter_object,
            log_to_console=False)
        result_image = itk.transformix_filter(
            moving_image,
            transform_parameters)

    if return_sitk:
        result_image = itk2sitk(result_image)
        result_image.SetDirection(original_moving_image.GetDirection())

    return result_image, transform_parameters

#for debugging parameter object
def print_itk_parameters(parameter_object):
    number_of_parameter_maps = parameter_object.GetNumberOfParameterMaps()

    for i in range(number_of_parameter_maps):
        print(i)
        pmap = parameter_object.GetParameterMap(i)
        print(f"ParameterMap {i}:")
        for key in pmap:
            print(f"  {key}: {pmap[key]}")
        print("\n")


####Simple ITK implementation is much slower than ITK implementation
def sitk_default_registration_args():
    sys.argv = ['ipykernel_launcher.py']

    parser = argparse.ArgumentParser(description="Default registration arguments")

    # Adding arguments
    parser.add_argument("--metric", type=str, default="NCC", choices=["CC", "MI", "MS", "NCC"],
                        help="Metric used for registration")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for NCC metric")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate for the optimizer")
    parser.add_argument("--num_iterations", type=int, default=500,
                        help="Number of iterations for the optimizer")
    parser.add_argument("--convergence_minimum_value", type=float, default=1e-6,
                        help="Convergence minimum value for the optimizer")
    parser.add_argument("--convergence_window_size", type=int, default=10,
                        help="Convergence window size for the optimizer")
    parser.add_argument('--number_of_resolutions', type=int, default=4,
                        help='Number of resolutions for the image pyramid.')
    parser.add_argument("--interpolator", default='sitkBSplineResamplerOrder3',
                        help="Type of resampler is an attribute of sitk")
    parser.add_argument("--default_pixel_value", type=int, default=-1024,
                        help="Background voxel values")
    parser.add_argument("--clip", type=tuple, default=(0, 100),
                        help="Used to clip registration images and apply transform after registration")

    # Parse the arguments
    args = parser.parse_args()
    return args

def sitk_apply_transform(fixed_image, moving_image, transform, args):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(args.interpolator)
    resampler.SetDefaultPixelValue(args.default_pixel_value)  # Set the default pixel value

    resampled_moving_image = resampler.Execute(moving_image)
    moved_image = sitk.Resample(moving_image, fixed_image, transform, args.interpolator, 0.0, moving_image.GetPixelID())
    return moved_image


def register_images_sitk(fixed_image, moving_image, args):
    # registers moving image to fixed image
    # for args see default_registration_args parser
    args.interpolator = getattr(sitk, args.interpolator)

    # convert to float for registration algoritm
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()

    # to construct a piramyd for multi level resolutions used shirnk factors and smoothing sigmas
    shrink_factors = [2 ** (args.number_of_resolutions - level - 1) for level in range(args.number_of_resolutions)]
    smoothing_sigmas = [2 ** (level) for level in reversed(range(args.number_of_resolutions))]

    # Configure the image pyramids for multi-resolution registration
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Similarity metric settings
    if args.metric == 'CC':
        registration_method.SetMetricAsCorrelation()
    elif args.metric == 'MI':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif args.metric == 'MS':
        registration_method.SetMetricAsMeanSquares()
    elif args.metric == 'NCC':
        registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=args.radius)

    # Interpolator settings
    registration_method.SetInterpolator(sitk.sitkBSplineResamplerOrder1)

    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=args.learning_rate,
        numberOfIterations=args.num_iterations,
        convergenceMinimumValue=args.convergence_minimum_value,
        convergenceWindowSize=args.convergence_window_size
    )

    # Registration settings
    registration_method.SetInitialTransform(
        sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    )

    if args.clip is not None:
        fx_img = sitk.Clamp(fixed_image, lowerBound=args.clip[0], upperBound=args.clip[1])
        mv_img = sitk.Clamp(fixed_image, lowerBound=args.clip[0], upperBound=args.clip[1])
    else:
        fx_img = fixed_image
        mv_img = moving_image

    # Optimize registration
    final_transform = registration_method.Execute(fx_img, mv_img)
    # Execute registration
    moved_image = sitk_apply_transform(fixed_image, moving_image, final_transform, args)
    moved_image = sitk.Cast(moved_image, sitk.sitkInt16)

    return moved_image, final_transform


def get_registration_matrix(fixed,
                            moving,
                            rp=None,
                            clip_range=None):
    if clip_range is not None:
        if isinstance(fixed, sitk.Image) and isinstance(moving, sitk.Image):
            fixed = sitk.Image(sitk.Clamp(fixed, lowerBound=clip_range[0], upperBound=clip_range[1]))
            fixed = sitk2ants(sitk.Cast(fixed, sitk.sitkFloat32))
            moving = sitk.Image(sitk.Clamp(moving, lowerBound=clip_range[0], upperBound=clip_range[1]))
            moving = sitk2ants(sitk.Cast(moving, sitk.sitkFloat32))

    if isinstance(fixed, sitk.Image):
        fixed = sitk2ants(sitk.Cast(fixed, sitk.sitkFloat32))

    if isinstance(moving, sitk.Image):
        moving = sitk2ants(sitk.Cast(sitk.Image(moving), sitk.sitkFloat32))

    if rp is None:
        rp = {'type_of_transform': 'TRSAA',
              'fix_bm': None,
              'mv_bm': None,
              'metric': 'mattes',
              'mask_all_stages': False,
              'default_value': 0,
              'interpolator': 'linear'  # or 'nearestNeighbor''bSpline'
              }
        print(rp)
        print('Using default registration params')

    mytx = ants.registration(fixed, moving,
                             type_of_transform=rp['type_of_transform'],
                             mask=rp['fix_bm'],
                             moving_mask=rp['mv_bm'],
                             mask_all_stages=rp['mask_all_stages'],
                             aff_metric=rp['metric'],
                             syn_metric=rp['metric']
                             )

    return mytx


def register_slab(fixed,
                  template_slab,
                  mv_slab=None,
                  interpolation='nearestNeighbor',
                  clip_range=(-50, 150),
                  slabname=None
                  ):
    """
    fixed: an sitk.Image in the target space
    template_slab: an sitk.Image ctp template slab to move (registration on cbf/cbv values vs fixed ct is not feasible)
    mv_slab: if defined (sitk.Image or list of sitk.Image) returns the coregisterd roi (segmentation) mask instead of template image
    interpolator: interpolation setting for mv_slab 'nearestNeighbor' or 'linear' are generally used

    returns registered template_slab, mv_slab, or list of mv_slabs
    """

    # Registration below uses default mattes and TRSAA registration settings
    tx = get_registration_matrix(
        sitk.Clamp(fixed, lowerBound=clip_range[0], upperBound=clip_range[1]),
        sitk.Clamp(template_slab, lowerBound=clip_range[0], upperBound=clip_range[1])
    )
    ants.write_transform(ants.read_transform(tx['fwdtransforms'][0]), d[f'ctp2ncct_reg_mat{slabname}'])

    if mv_slab is None:
        # returns template slab for debugging
        mv_out = ants.apply_transforms(sitk2ants(sitk.Cast(fixed, sitk.sitkFloat32)),
                                       sitk2ants(sitk.Cast(template_slab, sitk.sitkFloat32)),
                                       interpolator='linear',
                                       transformlist=tx['fwdtransforms'])
        mv_out = ants2sitk(mv_out)

    # mv_slab is image --> move that image to fixed space
    elif isinstance(mv_slab, sitk.Image):
        mv_out = ants.apply_transforms(sitk2ants(sitk.Cast(fixed, sitk.sitkFloat32)),
                                       sitk2ants(sitk.Cast(mv_slab, sitk.sitkFloat32)),
                                       interpolator=interpolator,
                                       transformlist=tx['fwdtransforms'])
        mv_out = ants2sitk(mv_out)

    # if mv_slab is list of images move each image to fixed space and return output
    elif isinstance(mv_slab, list):
        # multible slabs to move are present in mv_slab
        mv_out = []
        for mv in mv_slab:
            mvo = ants.apply_transforms(sitk2ants(sitk.Cast(fixed, sitk.sitkFloat32)),
                                        sitk2ants(sitk.Cast(mv, sitk.sitkFloat32)),
                                        interpolator=interpolator,
                                        transformlist=tx['fwdtransforms'])
            mv_out.append(ants2sitk(mvo))

    return mv_out



def register_slabs_dct(slab_dct,
                       fixed_img,
                       key_reg_slab=None,
                       rp=None,
                       clip_range=()):
    """
    slab_dct: key=image name, value= list of slabs
    fixed_img: sitk.Image in target space
    key_reg_slab: if not none, key in slab_dict which image slabs should be used for coregistration
                  all other image slabs are moved using registration params from this image
    rp: dict of registration parameters
    """
    if len(clip_range)>0:
        fixed_img = sitk.Clamp(fixed_img, lowerBound=clip_range[0], upperBound=clip_range[1])

    if key_reg_slab is not None:
        reg_slabs = slab_dct[key_reg_slab]

        # if multiple slabs (a list of slabs), register one by one
        if isinstance(reg_slabs, list):
            print('Registration of multiple slabs')
            out_dct = {k: [] for k in [*slab_dct.keys(), 'reg_mat']}
            for i in range(len(reg_slabs)):
                tx = get_registration_matrix(
                    fixed_img,
                    sitk.Clamp(reg_slabs[i], lowerBound=clip_range[0], upperBound=clip_range[1]
                              ) if len(clip_range)>0 else reg_slabs[i],
                    rp=rp
                )
                out_dct['reg_mat'].append(tx)

                for k, slab in slab_dct.items():
                    print('moving', i, k)
                    mv_slab = ants.apply_transforms(sitk2ants(sitk.Cast(fixed_img, sitk.sitkFloat32)),
                                                    sitk2ants(sitk.Cast(slab[i], sitk.sitkFloat32)),
                                                    interpolator='nearestNeighbor' if ('seg' in k or 'mask' in k) else 'linear',
                                                    transformlist=tx['fwdtransforms'])
                    out_dct[k].append(sitk.Cast(ants2sitk(mv_slab), sitk.sitkInt16))
        # if a single slab, it is easier; every value is a path
        else:
            print('Single slab registration')
            out_dct = {}
            tx = get_registration_matrix(
                fixed_img,
                sitk.Clamp(reg_slabs, lowerBound=clip_range[0], upperBound=clip_range[1]) if len(clip_range)>0 else reg_slabs,
                rp=rp
            )
            out_dct['reg_mat'] = tx

            for k, slab in slab_dct.items():
                print('moving', k)
                mv_slab = ants.apply_transforms(sitk2ants(sitk.Cast(fixed_img, sitk.sitkFloat32)),
                                                sitk2ants(sitk.Cast(slab, sitk.sitkFloat32)),
                                                interpolator='nearestNeighbor' if 'seg' in k else 'linear',
                                                transformlist=tx['fwdtransforms'])
                out_dct[k] = sitk.Cast(ants2sitk(mv_slab), sitk.sitkInt16)

    else:
        print('implement reg_slabs')

    return out_dct

def lateral_flip_coregister(image:sitk.Image, rp:dict, clip_range=(0,100), return_sitk=False):
    #flips and then coregisters image back
    #returns numpy by default
    flip = np2sitk(np.flip(sitk.GetArrayFromImage(image), axis=2), image)
    print('Registration')
    flipreg = ants_register(image,
                            flip,
                            rp=rp,
                            clip_range=clip_range)
    flipreg	= sitk.Cast(flipreg, sitk.sitkInt16)
    if not return_sitk:
        flipreg = sitk.GetArrayFromImage(flipreg)

    return flipreg

### tryout to get it to work on cmdline
# def itkelastix_default_arg_parser():
#     sys.argv = ['ipykernel_launcher.py']

#     parser = argparse.ArgumentParser(description="Elastix registration settings")

#     # Adding arguments with default values
#     parser.add_argument("--registration_type", type=str, default="rigid",
#                         choices=["rigid", 'affine'],
#                         help="Type of registration method")
#     parser.add_argument("--metric", type=str, default="AdvancedNormalizedCorrelation",
#                         choices=["AdvancedMattesMutualInformation", "NormalizedCorrelation", "AdvancedNormalizedCorrelation"],
#                         help="Metric used for registration")
#     parser.add_argument("--radius", type=int, default=2,
#                         help="Radius for metric, applicable if NCC metric is chosen")
#     parser.add_argument("--learning_rate", type=float, default=1.0,
#                         help="Learning rate for the optimizer")
#     parser.add_argument("--num_iterations", type=int, default=500,
#                         help="Number of iterations for the optimizer")
#     parser.add_argument("--convergence_minimum_value", type=float, default=1e-6,
#                         help="Convergence minimum value for the optimizer")
#     parser.add_argument("--convergence_window_size", type=int, default=10,
#                         help="Convergence window size for the optimizer")
#     parser.add_argument("--number_of_resolutions", type=int, default=4,
#                         help="Number of resolutions for the image pyramid.")
#     parser.add_argument("--interpolator", type=str, default="BSplineInterpolator", choices=["LinearInterpolator", "NearestNeighborInterpolator",],
#                         help="Type of interpolator")
#     parser.add_argument("--bspline_order", type=int, default=3,
#                         help="Order of the bspline interpolator")
#     parser.add_argument("--default_pixel_value", type=int, default=-1024,
#                         help="Default pixel value for outside image region")
#     parser.add_argument("--clip_values", type=int, nargs=2, default=(0, 100),
#                         help="Pair of min and max values used for clipping the images before registration")

#     return parser.parse_args()

# args = itkelastix_default_arg_parser()
# args

# # Set up the registration parameters
# parameter_object = itk.ParameterObject.New()
# parameter_map = parameter_object.GetDefaultParameterMap(args.registration_type, args.number_of_resolutions)
# parameter_map['Metric'] = [args.metric]
# if args.metric == 'AdvancedNormalizedCorrelation':
#     parameter_map['MetricRadius'] = [str(args.radius)]
# parameter_map['NumberOfResolutions'] = [str(args.number_of_resolutions)]
# parameter_map['MaximumNumberOfIterations'] = [str(args.num_iterations)]
# parameter_map['DefaultPixelValue'] = [str(args.default_pixel_value)]
# parameter_map['ResampleInterpolator'] = [args.interpolator]
# if 'bspline' in args.interpolator.lower():
#     parameter_map['BSplineInterpolationOrder'] = [str(args.bspline_order)]
# parameter_map['WriteResultImage'] = ['false']

# parameter_object.AddParameterMap(parameter_map)

# # Perform registration
# result_image, result_transform_parameters = itk.elastix_registration_method(fixed_image = fixed_image,
#                                                                             moving_image=moving_image,
#                                                                             parameter_object=parameter_object)
# itk.elastix_registration_method