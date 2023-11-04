
import SimpleITK as sitk
import sys
import argparse

def default_registration_args():
    sys.argv = ['ipykernel_launcher.py']

    parser = argparse.ArgumentParser(description="Default registration arguments")

    # Adding arguments
    parser.add_argument("--metric", type=str, default="NCC", choices=["CC", "MI", "MS", "NCC"],
                        help="Metric used for registration")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for NCC metric")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate for the optimizer")
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="Number of iterations for the optimizer")
    parser.add_argument("--convergence_minimum_value", type=float, default=1e-6,
                        help="Convergence minimum value for the optimizer")
    parser.add_argument("--convergence_window_size", type=int, default=10,
                        help="Convergence window size for the optimizer")
    parser.add_argument('--number_of_resolutions', type=int, default=4,
                        help='Number of resolutions for the image pyramid.')
    parser.add_argument("--interpolator", default=sitk.sitkBSplineResamplerOrder3,
                        help="Type of resampler is an attribute of sitk")
    parser.add_argument("--default_pixel_value", type=int, default=-1024,
                        help="Background voxel values")
    parser.add_argument("--clip", type=tuple, default=(0, 100),
                        help="Used to clip registration images and apply transform after registration")

    # Parse the arguments
    args = parser.parse_args()
    return args

def apply_transform(fixed_image, moving_image, transform, args):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(args.interpolator)
    resampler.SetDefaultPixelValue(args.default_pixel_value)  # Set the default pixel value

    resampled_moving_image = resampler.Execute(moving_image)
    moved_image = sitk.Resample(moving_image, fixed_image, transform, args.interpolator, 0.0, moving_image.GetPixelID())
    return moved_image


def register_images(fixed_image, moving_image, args):
    # registers moving image to fixed image
    # for args see default_registration_args parser

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
    moved_image = apply_transform(fixed_image, moving_image, final_transform, args)
    moved_image = sitk.Cast(moved_image, sitk.sitkInt16)

    return moved_image, final_transform

