import SimpleITK as sitk
import ants
import numpy as np
def sitk2ants(sitk_image):
    #convert simpleITK image to ants format
    # First, convert the SimpleITK image to a numpy array
    np_array = sitk.GetArrayFromImage(sitk_image)
    np_array = np.transpose(np_array, (2, 1, 0))
    # Use the origin, spacing, and direction from the SimpleITK image
    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()
    direction = np.array(sitk_image.GetDirection()).reshape(sitk_image.GetDimension(), sitk_image.GetDimension())

    # Convert the numpy array to an ANTs image
    ants_image = ants.from_numpy(np_array, origin=origin, spacing=spacing, direction=direction)
    return ants_image


def ants2sitk(ants_image):
    """
    Convert an ANTs image to a SimpleITK image.

    Parameters:
    ants_image (ants.Image): The ANTs image to convert.

    Returns:
    SimpleITK.Image: The converted SimpleITK image.
    """
    # Convert the ANTs image to a NumPy array
    np_image = ants_image.numpy().astype(np.int16)
    np_image = np.transpose(np_image, (2, 1, 0))
    # Create a SimpleITK image from the NumPy array
    sitk_image = sitk.GetImageFromArray(np_image)
    # Set the spacing, origin, and direction from the ANTs image to the SimpleITK image
    sitk_image.SetSpacing(ants_image.spacing)
    sitk_image.SetOrigin(ants_image.origin)
    sitk_image.SetDirection(ants_image.direction.flatten())
    return sitk_image

#convert ants image to preferred integer type (reduces size for storage)
def ants_astype(img,np_type=np.int16):
    cloned_image = ants.image_clone(img).numpy().astype(np_type)
    img = ants.from_numpy(cloned_image,
                          origin=img.origin,
                          spacing=img.spacing,
                          direction=img.direction)
    return img


###ants register function


