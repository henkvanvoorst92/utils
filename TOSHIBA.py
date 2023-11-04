import pandas as pd
import pydicom
import os
import numpy as np
import SimpleITK as sitk
import sys
import re
from tqdm import tqdm
from utils.utils import rtrn_np, list_files, try_t_str, all_tag_values, exists, is_ascending


def create_mdct(dcm):
    #extracts all metadata names and IDs
    mdct = {}
    #iterating over items does not work for v
    for k in dcm.keys():
        try:
            n = re.subn('[ ]()', '', str(dcm[k].name))[0]
            if n=='PixelData':
                continue
            mdct[n] = k
        except:
            print('Does not work:', dcm[k])
    return mdct

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


def toshiba_get_metadata(p_dcmdir: str, time_dcm_tag=None):
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
        mdct = create_mdct(dcm)
        row = all_tag_values(dcm, mdct, stringconvert=True)
        tmp = pd.DataFrame(row).T
        tmp.columns = list(mdct.keys())
        tmp['Volno'] = Volno
        tmp['Volstr'] = Volstr
        tmp['dcmfile'] = file
        scan_mdata.append(tmp)
    mdata = pd.concat(scan_mdata).reset_index(drop=True)

    # perform sanity check of time tag to see if files are
    # sequential in time
    if time_dcm_tag is not None:  # preferred: 'AcquisitionDateTime'
        times = np.array([try_t_str(t, None) for t in mdata[time_dcm_tag ].values])
        ascends = is_ascending(times)
        if not ascends:
            # create new index number of timefram (Volno)
            mdata['newtimevar'] = times
            mdata = mdata.sort_values(by='newtimevar')
            mdata['Volno'] = mdata.index + 1
            print('Reordered the time sequence for:', Volno,file)

    return mdata

#combine metadata

if __name__=='__main__':
    root = '/media/hvv/ec2480e5-6c18-468c-b971-5271432b386d/hvv/Toshiba_sc'
    list_files(root)
    for ID in os.listdir(root):
        pid = os.path.join(root,ID)
        p_dcmdir = os.path.join(pid,'DCM')
        dcm_files = os.listdir(p_dcmdir)
        break




