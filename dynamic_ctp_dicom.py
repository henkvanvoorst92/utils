import SimpleITK as sitk
import numpy as np
import pydicom
import os
import sys
#create a dummy dataset 128x128x16x50
sys.path.append('..')
from utils.dicomutils import sitk2generic_ct
import datetime
from utils.preprocess import sitk_flip_AP

def preprocess_ctp(input_ctp):
    # changes dimensions according to sorens requirements for example_4d_sitk_to_DICOM
    vol3d = input_ctp[:, :, :, 0]

    myarr = sitk.GetArrayFromImage(input_ctp)
    myarr = np.transpose(myarr, [1, 2, 3, 0]).astype(np.int16)

    # convert to SimpleITK format
    dyn4d = sitk.GetImageFromArray(myarr, isVector=True)
    dyn4d.CopyInformation(vol3d)

    return dyn4d


def example_4d_sitk_to_DICOM(sitkimage,opfolder):

    arr4d = sitk.GetArrayViewFromImage(sitkimage)
    spacing = sitkimage.GetSpacing()

    shared_header = {'SeriesInstanceUID': pydicom.uid.generate_uid(),
                     'StudyInstanceUID': pydicom.uid.generate_uid(),
                     'FrameOfReferenceUID': pydicom.uid.generate_uid(),
                     'Rows': arr4d.shape[1],
                     'Columns': arr4d.shape[2],
                     'PixelSpacing': ["{:2.8f}".format(spacing[1]), "{:2.8f}".format(spacing[0])],
                     # between rows / cols
                     'PatientID': 'Noname',
                     'PatientName': 'Noname',
                     'StudyDescription': 'StrokeStudy',
                     'SeriesDescription': 'CTP',
                     'Modality': 'CT',
                     'RescaleSlope': 1,
                     'RescaleIntercept': -1024,
                     'SeriesNumber': '100'}

    nframes = arr4d.shape[3]
    nslices = arr4d.shape[2]
    start_time = datetime.datetime(year=2024,month=1,day=1,hour=12,minute=00,second=0)
    current_datetime = start_time
    delta_t = 1.00
    IOP = sitkimage.GetDirection()
    instance_number_offset = 1
    for iframe in range(nframes):
        current_frame_img = sitk.GetImageFromArray(arr4d[:,:,:,iframe])
        current_frame_img.CopyInformation(sitkimage)


        shared_header["AcquisitionDate"] = current_datetime.strftime("%Y%m%d")
        shared_header["AcquisitionTime"] = current_datetime.strftime("%H%M%S.%f")
        shared_header["SeriesDate"] = start_time.strftime("%Y%m%d")
        shared_header["SeriesTime"] = start_time.strftime("%H%M%S")
        shared_header["StudyDate"] = start_time.strftime("%Y%m%d")
        shared_header["StudyTime"] = start_time.strftime("%H%M%S")


        sitk2generic_ct(current_frame_img, ct_outfolder=opfolder,instance_number_start=instance_number_offset,seriesheader=shared_header)

        instance_number_offset += arr4d.shape[3]
        current_datetime = start_time + datetime.timedelta(seconds=delta_t)


if __name__ == "__main__":

    i = 0
    for f in tqdm(os.listdir(p_in)):
        ID = f.split('_')[0]

        dir_out = os.path.join(p_out, ID)
        if os.path.exists(dir_out):
            continue

        print(ID)
        file = os.path.join(p_in, f)
        ctp = sitk.ReadImage(file)
        # ctp = sitk.JoinSeries([sitk.Cast(sitk_flip_AP(ctp[:,:,:,i]), sitk.sitkInt16) for i in range(ctp.GetSize()[-1])])
        os.makedirs(dir_out, exist_ok=True)
        pp_ctp = preprocess_ctp(ctp)
        example_4d_sitk_to_DICOM(pp_ctp, dir_out)
        i += 1
        if i > 3:
            break






    # opfolder = "DCM"
    #
    # myarr = np.zeros( (16,128,128,50),dtype=np.int16)
    #
    # #label each frame with it's index, start at 0
    # for iframe in range(myarr.shape[3]):
    #     myarr[:,:,:,iframe] = iframe
    #
    # #convert to SimpleITK format
    #
    # dyn4d = sitk.GetImageFromArray(myarr,isVector=True)
    #
    # assert dyn4d.GetNumberOfComponentsPerPixel() == 50
    # if not os.path.exists(opfolder):
    #     os.makedirs(opfolder)
    sitk.WriteImage(dyn4d,os.path.join(opfolder,"dyn.nii"))

    example_4d_sitk_to_DICOM(dyn4d,"DCM")


#so now we have a 4D sitk image as a starting point. We will split it up again now so we can call the 3D writer multiple times
