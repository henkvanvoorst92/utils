
import os
import sys
import SimpleITK as sitk
def write_as_nnunet(IMG, GT, p_dir, ID):
    if IMG is not None:
        p_img_new = os.path.join(p_dir ,'imagesTr')
        if not os.path.exists(p_img_new):
            os.makedirs(p_img_new)
        sitk.WriteImage(IMG ,os.path.join(p_img_new, ID+'_0000.nii.gz'))

    if GT is not None:
        p_gt_new = os.path.join(p_dir , 'labelsTr')
        if not os.path.exists(p_gt_new):
            os.makedirs(p_gt_new)
        sitk.WriteImage(GT ,os.path.join(p_gt_new, ID+'.nii.gz'))
