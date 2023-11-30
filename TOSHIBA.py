import pandas as pd
import pydicom
import os
import numpy as np
import SimpleITK as sitk
import itk
from tqdm import tqdm
from utils.utils import rtrn_np, list_files, exists, is_ascending
from utils.utils import try_t_str, all_tag_values, get_tagnos_tagnames, timedelta2str, datetime2str
from utils.registration import itk_register
from utils.preprocess import sitk_add_tags

def toshiba_ID(pdir:str):

    if not os.path.exists(pdir):
        raise ValueError ('Not an existing path:', pdir)

    out = []
    for dr in os.listdir(pdir):
        volstr = os.path.splitext(dr)[0].split('_')[-1]
        try:
            volno = int(volstr)
        except:
            raise ValueError ('No integer:',ID)
        out.append([volno, volstr, os.path.join(pdir,dr)])

    return pd.DataFrame(out, columns=['Volno', 'Volstr', 'path']).sort_values(by='Volno').reset_index(drop=True)


# strip all dicom tags you can find in a single CTP frame
def toshiba_all_tag_keys_values(dcm):
    # return a list of key and values from Thosiba dicom tags
    # first extract the easy to access global tags
    tagname2no, tagno2name = get_tagnos_tagnames(dcm)
    row_tags = all_tag_values(dcm, tagno2name, stringconvert=False)
    cols = list(tagname2no.keys())
    # now extract the PerFrameFunctionalGroupsSequence and SharedFunctionalGroupsSequence tags
    # see also: https://stackoverflow.com/questions/74776837/pydicom-returns-keyerror-even-though-field-exists
    subdcm = dcm['PerFrameFunctionalGroupsSequence'][10]
    for subkey in subdcm.keys():
        subsubdcm = subdcm[subkey][0]
        tagname2no, tagno2name = get_tagnos_tagnames(subsubdcm)
        r = all_tag_values(subsubdcm, tagno2name, stringconvert=True)
        row_tags.extend(r)
        cols.extend(list(tagname2no.keys()))

    subdcm = dcm['SharedFunctionalGroupsSequence'][0]
    for subkey in subdcm.keys():
        subsubdcm = subdcm[subkey][0]
        tagname2no, tagno2name = get_tagnos_tagnames(subsubdcm)
        r = all_tag_values(subsubdcm, tagno2name, stringconvert=True)
        row_tags.extend(r)
        cols.extend(list(tagname2no.keys()))

    row_tags = np.array(row_tags)
    cols = np.array(cols)

    isna_vec = np.vectorize(pd.isna)
    na_mask = isna_vec(row_tags)

    row_tags = row_tags[~na_mask]
    cols = cols[~na_mask]

    return cols, row_tags


# put all metadatafrom all timeframes
def toshiba_timeframes_metadata(p_dcmdir: str, time_dcm_tag=None):
    """
    input: path to directory with dicom files (p_dcmdir)
    optional time_dcm_tag: dicom tag representing acquisition time
    Iterates over sorted (by volume number) dicom files per ctp time series
    each dicom file is a Toshiba scan volume (dims=xyz)
    if time_dcm_tag: do sanity check to see if file names are
    ordered ascending in time
    returns a pd dataframe with all dicom tag metadata
    """
    df = toshiba_ID(p_dcmdir)
    # process all scans in the time series
    scan_mdata = []
    for __, (Volno, Volstr, file) in df.iterrows():
        dcm = pydicom.dcmread(file, stop_before_pixels=True)
        C, R = toshiba_all_tag_keys_values(dcm)
        tmp = pd.DataFrame(data=R, index=C).T
        tmp['Volno'] = Volno
        tmp['Volstr'] = Volstr
        tmp['dcmfile'] = file
        scan_mdata.append(tmp)
    mdata = pd.concat(scan_mdata).reset_index(drop=True)

    # perform sanity check of time tag to see if files are
    # sequential in time
    if time_dcm_tag is not None:  # preferred: 'AcquisitionDateTime'
        times = np.array([try_t_str(t, None) for t in mdata[time_dcm_tag].values])
        ascends = is_ascending(times)
        if not ascends:
            # create new index number of timefram (Volno)
            mdata['newtimevar'] = times
            mdata = mdata.sort_values(by='newtimevar')
            mdata['Volno'] = mdata.index + 1
            print('Reordered the time sequence for:', Volno, file)

    return mdata

def process_ctp_frames(mdata,
                       param_file=None,
                       sloc_pid = None,
                       clip_bounds=None,
                       register=True
                       ):

    if sloc_pid is not None:
        sloc_org_nii = os.path.join(sloc_pid, 'NII')
        sloc_reg_nii = os.path.join(sloc_pid, 'REG')
        sloc_trans_files = os.path.join(sloc_pid, 'transform_params')
    else:
        sloc_org_nii = None
        sloc_reg_nii = None
        sloc_trans_files = None

    minVolno = mdata['Volno'].min()
    out4D = []
    out4D_reg = []
    for no, file in tqdm(zip(list(mdata['Volno']), list(mdata['dcmfile']))):
        # print(no,file, type(file), os.path.exists(file))
        img = sitk.ReadImage(file)
        img = sitk.Cast(sitk.Clamp(img, lowerBound=-1024, upperBound=1000), sitk.sitkInt16)
        out4D.append(img)
        if sloc_org_nii is not None:
            exists(sloc_org_nii)
            sitk.WriteImage(img, os.path.join(sloc_org_nii, str(no) + '.nii.gz'))
        if register:
            # store first volume to register to
            if no == minVolno:
                start_img = img
                out4D_reg.append(img)
                if sloc_reg_nii is not None:
                    exists(sloc_reg_nii)
                    sitk.WriteImage(start_img, os.path.join(sloc_reg_nii, str(no) + '.nii.gz'))
            else:
                reg_img, transforms = itk_register(start_img, img, param_file, clip_bounds=clip_bounds)
                reg_img = sitk.Cast(reg_img, sitk.sitkInt16)
                out4D_reg.append(reg_img)
                # store registered file
                if sloc_reg_nii is not None:
                    exists(sloc_reg_nii)
                    sitk.WriteImage(reg_img, os.path.join(sloc_reg_nii, str(no) + '.nii.gz'))
                # store reg params
                if sloc_trans_files is not None:
                    exists(sloc_trans_files)
                    transform_file = os.path.join(sloc_trans_files, 'transform_{}.txt'.format(no))
                    itk.ParameterObject.New().WriteParameterFile(transforms, transform_file)

    str_datetime = str(datetime2str(mdata['AcquisitionDateTime'].to_list()))
    str_t_diff = str(timedelta2str(mdata['TimeDifference'].to_list()))

    #add metadata to 4D sequence
    if len(out4D)>2:
        out4D = sitk.JoinSeries(out4D)
        out4D.SetMetaData('AcquisitionDateTime', str_datetime)
        out4D.SetMetaData('TimeDifference', str_t_diff)
        out4D = sitk_add_tags(out4D, mdata, tags=['ExposureinmAs', 'KVP', 'CTDIvol'])
        if sloc_pid is not None:
            sitk.WriteImage(out4D, os.path.join(sloc_pid,'CTP.nii.gz'))

    if len(out4D_reg)>2:
        out4D_reg = sitk.JoinSeries(out4D_reg)
        out4D_reg.SetMetaData('AcquisitionDateTime', str_datetime)
        out4D_reg.SetMetaData('TimeDifference', str_t_diff)
        out4D_reg = sitk_add_tags(out4D_reg, mdata, tags=['ExposureinmAs', 'KVP', 'CTDIvol'])
        if sloc_pid is not None:
            sitk.WriteImage(out4D_reg, os.path.join(sloc_pid,'CTP_reg.nii.gz'))

    return out4D, out4D_reg


if __name__=='__main__':
    root = '/media/hvv/ec2480e5-6c18-468c-b971-5271432b386d/hvv/Toshiba_sc'
    list_files(root)
    for ID in os.listdir(root):
        pid = os.path.join(root,ID)
        p_dcmdir = os.path.join(pid,'DCM')
        dcm_files = os.listdir(p_dcmdir)
        break




