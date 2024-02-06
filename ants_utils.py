import SimpleITK as sitk
import ants
import numpy as np

def sitk2ants(sitk_image):
    """
    Convert a SimpleITK image to an ANTs image.

    Parameters:
    sitk_image (SimpleITK.Image): The SimpleITK image to convert.

    Returns:
    ants.Image: The converted ANTs image.
    """
    # Convert the SimpleITK image to a NumPy array
    np_image = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

    # Create an ANTs image from the NumPy array
    ants_image = ants.from_numpy(np_image)

    # Set the spacing, origin, and direction from the SimpleITK image to the ANTs image
    ants_image.set_spacing(sitk_image.GetSpacing())
    ants_image.set_origin(sitk_image.GetOrigin())
    ants_image.set_direction(np.array(sitk_image.GetDirection()).reshape(sitk_image.GetDimension(), sitk_image.GetDimension()))

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


