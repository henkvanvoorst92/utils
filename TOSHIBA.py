
#### used for processing original Toshiba CTP data
#probably a lot of overlap with other data

import pandas as pd
import pydicom
import os
import numpy as np
import SimpleITK as sitk
import shutil
import argparse
from tqdm import tqdm
from totalsegmentator.python_api import totalsegmentator

from utils.utils import list_files, exists, is_ascending, is_notebook, compute_time_differences
from utils.utils import try_t_str, all_tag_values, get_tagnos_tagnames, timedelta2str, datetime2str
from utils.registration import ants_register
from utils.preprocess import sitk_add_tags,  ctp_exposure_weights, np2sitk, dcm2sitk, dcm2niix


#from utils.registration import itk_register
#from utils.torch_utils import rtrn_np

def toshiba_ID(pdir:str):
    #used for sorens initial data
    if not os.path.exists(pdir):
        raise ValueError ('Not an existing path:', pdir)

    out = []
    for dr in os.listdir(pdir):
        volstr = os.path.splitext(dr)[0].split('_')[-1]
        try:
            volno = int(volstr)
        except:
            raise ValueError ('No integer:',dr)
        out.append([volno, volstr, os.path.join(pdir,dr)])

    return pd.DataFrame(out, columns=['Volno', 'Volstr', 'path']).sort_values(by='Volno').reset_index(drop=True)

def toshiba_ID2(pdir: str):
    # used for melbourne data
    if not os.path.exists(pdir):
        raise ValueError('Not an existing path:', pdir)

    out = []
    for ix, dr in enumerate(os.listdir(pdir)):
        if '.dbi' in dr:
            continue
        volstr = dr
        try:
            volno = int(volstr.replace('IM', ''))
        except:
            volno = ix
        out.append([volno, volstr, os.path.join(pdir, dr)])

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
    try:
        row_tags = np.array(row_tags)
        cols = np.array(cols)

        isna_vec = np.vectorize(pd.isna)
        na_mask = isna_vec(row_tags)

        row_tags = row_tags[~na_mask]
        cols = cols[~na_mask]
        output = (cols,row_tags)
    except:
        output = pd.DataFrame(index=row_tags).reset_index().T
        output.columns = cols

    return output

# put all metadatafrom all timeframes together
def toshiba_timeframes_metadata(p_dcmdir: str, time_dcm_tag=None, toshiba_ID=toshiba_ID):
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

        tmp = toshiba_all_tag_keys_values(dcm)
        if isinstance(tmp, tuple):
            tmp = pd.DataFrame(data=tmp[1], index=tmp[0]).T
        elif isinstance(tmp, pd.DataFrame) or isinstance(tmp, pd.Series):
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


### scripts for new melbourne data


def ctp_all_timeframes_averaging(ctp: sitk.Image or str,
                                 exposures: np.ndarray,
                                 global_avg: list or np.ndarray = ['avg', 50, 75, 90],
                                 sav_dir=None,
                                 nnunet_folder=None,
                                 ID=None,
                                 addname='',
                                 overwrite=False):
    out = None
    if isinstance(ctp, str):
        ctp = sitk.ReadImage(ctp)

    # for saving in the patient ID folder
    if sav_dir is not None:
        os.makedirs(sav_dir, exist_ok=True)
        save = True
    else:
        save = False

    # for directly saving in the nnUnet vesselseg inference folder
    if nnunet_folder is not None:
        os.makedirs(nnunet_folder, exist_ok=True)
        save_nnunet = True
    else:
        save_nnunet = False

    # assert if the files are not already present
    # then the analyses can be skipped
    has_image = []
    for v in global_avg:
        if v == 'avg':
            vname = v
        elif isinstance(v, int):
            vname = f'pct{v}'
        if ID is not None:
            filename = ID + f'-{vname}' + addname
        else:
            filename = vname + addname
        if save:
            p_sav = os.path.join(sav_dir, filename + '.nii.gz')
            exsts = os.path.exists(p_sav)
        if save_nnunet:
            p_sav_nnunet = os.path.join(nnunet_folder, filename + '_0000.nii.gz')
            exsts *= os.path.exists(p_sav_nnunet)
        has_image.append(exsts)

    if not all(has_image) or overwrite:
        # only perform analyses if images are not there
        # or if overwriting is on

        wexp = ctp_exposure_weights(exposures)
        ctp_np = np.clip(sitk.GetArrayFromImage(ctp), -1024, 2000).astype(type(wexp[0]))
        ctp_np *= wexp[:, np.newaxis, np.newaxis, np.newaxis]

        # write something to get average and percentile images
        out = {}
        for v in global_avg:
            if v == 'avg':
                img = ctp_np.mean(axis=0)
                vname = v
            elif isinstance(v, int):
                img = np.percentile(ctp_np, v, axis=0)
                vname = f'pct{v}'
            else:
                print('Unknown global_avg value:', v)
                continue

            img = np2sitk(img, ctp[:, :, :, 0])
            out[vname] = img

            if ID is not None:
                filename = ID + f'-{vname}'
            else:
                filename = vname
            if len(addname) > 0:
                filename += addname

            if save:
                p_sav = os.path.join(sav_dir, filename + '.nii.gz')
                sitk.WriteImage(img, p_sav)
            if save_nnunet:
                p_sav_nnunet = os.path.join(nnunet_folder, filename + '_0000.nii.gz')
                # sitk.WriteImage(img, p_sav_nnunet)
                shutil.copy2(p_sav,p_sav_nnunet)

    return out

def register_ctp_frames(ctp,
                        rp,
                        p_ctp_reg=None):
    """
    Function registers all frames to the first
    ctp: can be an sitk.Image or path to .nii.gz sitkImage, cpt has 4 dims (3D+time)
    rp: dict with ants registration params
    p_ctp_reg: path to store registered CTP 4D cube

    Return 4D ctp as sitk.Image
    """
    if isinstance(ctp, str):
        # if ctp is a path
        ctp = sitk.ReadImage(ctp)

    frames = []
    for i in tqdm(range(ctp.GetSize()[-1])):
        frame = sitk.Cast(sitk.Clamp(ctp[:, :, :, i], lowerBound=-1024, upperBound=2000), sitk.sitkInt16)
        if i == 0:
            reference_frame = sitk.Image(frame)
            frames.append(reference_frame)
        else:
            # register each frame to the first using rigid reg
            reg_frame = ants_register(reference_frame,
                                      frame,
                                      rp=rp,
                                      addname=f'{i}',
                                      p_transform=rp['p_transforms'],
                                      clip_range=rp['clip_range'])
            frames.append(reg_frame)

    ctp_reg = sitk.JoinSeries(frames)
    if p_ctp_reg is not None:
        sitk.WriteImage(ctp_reg, p_ctp_reg)
    return ctp_reg

def get_ctp_values_inside_mask(ctp_np,bm_np):
    out = []
    for i in range(ctp_np.shape[0]):
        out.append(ctp_np[i][bm_np==1])
    return np.vstack(out)

def mean_brain_hu_per_frame(ctp: sitk.Image or str,
                            bm: sitk.Image or str,
                            ):
    if isinstance(bm, str):
        bm = sitk.Cast((sitk.ReadImage(bm) == 90) * 1, sitk.sitkInt16)
    bm_np = sitk.GetArrayFromImage(bm)

    if isinstance(ctp, str):
        ctp = sitk.ReadImage(ctp)
    ctp_np = sitk.GetArrayFromImage(ctp)

    vals = get_ctp_values_inside_mask(ctp_np, bm_np)
    mean_per_frame = vals.mean(axis=1)  # pm add these values to the mdata file (both excel and pickle)

    return mean_per_frame

def ctp2cta(ctp: sitk.Image or str,
               exposures: np.ndarray or list,
               margin: int,
               ID: str,
               dir_sav: str = None,
               nnunet_folder: str = None,
               addname: str = '',
               cta_ixmax: int = None,
               all_ctas = False,
               overwrite: bool = False):
    """
    returns dict of all frames averaged over neighboring margin frames using exposure weights

    exposures: used to weight the frames, some protocols have different brightnesses of frames
    margin: number of pre- post- frames
    ID: str representing ID for naming files
    dir_sav: directory to save the images, if None no saving performed
    nnunet_folder: directly saves images to nnunet style folder for inference
    addname: file naming to add
    cta_ixmax: ix representing ctp frame number of peak arterial cta --> stores file separately
    overwrite: if True overwrites files
    """

    out = None
    if isinstance(ctp, str):
        ctp = sitk.ReadImage(ctp)
    ctp_np = sitk.GetArrayFromImage(ctp)

    # for saving in the patient ID folder
    if dir_sav is not None:
        save = True
    else:
        save = False

    # for directly saving in the nnUnet vesselseg inference folder
    if nnunet_folder is not None:
        os.makedirs(nnunet_folder, exist_ok=True)
        save_nnunet = True
    else:
        save_nnunet = False

    sid = os.path.join(dir_sav, 'cta_frames')
    os.makedirs(sid, exist_ok=True)

    # check if frames exist
    if all_ctas:
        done_ixs = [int(f.replace(ID + '_', '').replace(addname, '').replace('.nii.gz', '')) for f in os.listdir(sid)]
    else:
        done_ixs = []
    # iterate over all timeframes
    out = {}
    for i in range(margin, ctp_np.shape[0] - margin):
        if i in done_ixs and not overwrite:
            continue
        #if not all_ctas, only run if cta_ixmax==1
        if cta_ixmax is not None:
            if i!=cta_ixmax and not all_ctas:
                continue
        # compute mean weighted for exposure (mAs) and time distance
        wexp = ctp_exposure_weights(exposures[i - margin:i + margin + 1])
        cta_np = np.clip(ctp_np[i - margin:i + margin + 1], -1024, 2000)
        cta_np = (cta_np * wexp[:, np.newaxis, np.newaxis, np.newaxis]).mean(axis=0)
        cta = np2sitk(cta_np, ctp[:, :, :, 0])
        out[i] = cta

        p_sav = os.path.join(sid, f'{ID}_{i}{addname}.nii.gz')
        if save:
            sitk.WriteImage(cta, p_sav)
            if save_nnunet:
                p_sav_nnunet = os.path.join(sid, f'{ID}_{i}{addname}.nii.gz')
                shutil.copy2(p_sav, p_sav_nnunet)

            if cta_ixmax is not None:
                if i == cta_ixmax:
                    f_cta = os.path.join(dir_sav, f'{ID}_genCTA{addname}.nii.gz')
                    shutil.copy2(p_sav, f_cta)
                    if save_nnunet:
                        p_sav_nnunet = os.path.join(sid, f'{ID}_genCTA{addname}.nii.gz')
                        shutil.copy2(p_sav, p_sav_nnunet)

    return out


def init_args(args=None):
    # Create the parser
    parser = argparse.ArgumentParser(
        description='Process CTP data with options for time averaging, mask ROI segmentation, CTA sequence generation, and more.')

    # Required positional arguments
    parser.add_argument('--root', type=str, default=None,
                        help='Main directory to store all the data.')
    parser.add_argument('--ctp_df', type=str, default=None,
                        help='xslx file with several required columns, used for finding and parsing scans')
    """
    Required columns for ctp_df:
    'ID_new_dupl': cleaned ID ready to create folder names
    'dir': directory with the ctp scan dicoms
    'AcquisitionDateTime': timestamp for each dicom representing a ctp frame volume
    """
    # Optional arguments
    parser.add_argument('--time_averages', nargs='*',
                        type=str or int, default=['avg', 50, 75, 90],
                        help="list to average or take percentile across ")
    parser.add_argument('--mask_roi_subset', nargs='*', type=str,
                        default=['brain', 'skull'],
                        help="Input for totalsegmentator to segment brain and skull in the first frame. Provide as space-separated list (e.g., 'brain skull'). Skipped if not provided.")

    parser.add_argument('--all_ctas', action='store_true',
                        help='If provided, generates new CTP sequence with averaging over window.')
    parser.add_argument('--margin', type=int, default=1,
                        help='N-frames before and after the current frame to aggregate using Exposure of each frame. Skipped if None provided.')
    parser.add_argument('--prepare_nnunet', action='store_true',
                        help='If provided, copies time averages and averages over sliding window (using margin).')
    parser.add_argument('--overwrite', action='store_true',
                        help='If provided, overwrites existing files.')
    parser.add_argument('--register', action='store_true',
                        help='If provided, performs registration.')

    # Registration parameters as a group
    reg_group = parser.add_argument_group('Registration Parameters')
    reg_group.add_argument('--type_of_transform', type=str,
                           default='Rigid',
                           help='Type of transformation for registration using ants. Default is "Rigid".')
    reg_group.add_argument('--metric', type=str,
                           default='mattes', help='Metric for registration using ants. Default is "mattes".')
    reg_group.add_argument('--default_value', type=int, default=-1024,
                           help='Default value for registration. Default is -1024.')
    reg_group.add_argument('--clip_range', nargs=2, type=int,
                           default=(-100, 200),
                           help='Clip range for registration. Provide as two space-separated values (e.g., -100 200).')

    if is_notebook():
        print("Detected notebook environment, using default argument values.")
        return parser.parse_args([])
    else:
        return parser.parse_args(args)


if __name__=='__main__':

    """
    Required arguments to run this script:
    ctp_df: should contain rows with columns:
        'ID_new_dupl': cleaned ID ready to create folder names
        'dir': directory with the ctp scan dicoms
        'AcquisitionDateTime': timestamp for each dicom representing a ctp frame volume
    root: main directory to store all the data
    time_averages: if list processes all ctp frames as average or percentiles 
                   across time to one image, skipped if list
    mask_roi_subset: input for totalsegmentator to segment brain and skull in 
                     the first frame, skipped if None
    all_ctas: if True generates new CTP sequence with averaging over window
    margin: n-frames before and after current frame to aggregate using Exposure of each frame
            if None, this step is skipped
    prepare_nnunet: copies time_averages, and averages over sliding window (using margin)
    overwrite: if True overwrites existing files
    """

    args = init_args()

    root = args.root
    ctp_df = pd.read_excel(args.ctp_df)
    p_sav = os.path.join(root, 'processed_data')

    time_averages = args.time_averages
    mask_roi_subset = args.mask_roi_subset
    margin = args.margin
    all_ctas = args.all_ctas
    register = args.register
    prepare_nnunet = args.prepare_nnunet
    overwrite = args.overwrite

    registration_params = {'type_of_transform': args.type_of_transform,
                           'fix_bm': None,
                           'mv_bm': None,
                           'metric': args.metric,
                           'mask_all_stages': False,
                           'default_value': args.default_value,
                           'clip_range': args.clip_range,
                           }

    if prepare_nnunet:
        nnunet_folder = os.path.join(root, 'nnunet_scans')
    else:
        nnunet_folder = None

    for dir_data, row in tqdm(ctp_df.iterrows()):
        rp = registration_params.copy()
        ctp, mdata, ctp_reg = None, None, None

        ID = row['ID_new_dupl']
        pid = os.path.join(p_sav, ID)

        pid_nii = os.path.join(pid, 'NII')
        os.makedirs(pid_nii, exist_ok=True)

        if register:
            pid_reg = os.path.join(pid, 'REG')
            os.makedirs(pid_reg, exist_ok=True)
            rp['p_transforms'] = os.path.join(pid, 'transforms')
            os.makedirs(rp['p_transforms'], exist_ok=True)

        print(ID)

        filename = ID + '_CTP'
        p_ctp = os.path.join(pid, filename)
        if not os.path.exists(p_ctp + '.nii.gz') and os.path.exists(row['dir']):
            dcm2niix(filename, row['dir'], pid)

            mdata = toshiba_timeframes_metadata(p_dcmdir=row['dir'],
                                                time_dcm_tag='AcquisitionDateTime',
                                                toshiba_ID=toshiba_ID2)

            mdata['OrgAcquisitionDateTime'] = mdata['AcquisitionDateTime']
            mdata['AcquisitionDateTime'] = [try_t_str(t, None) for t in mdata['AcquisitionDateTime']]
            mdata['TimeDifference'] = compute_time_differences(list(mdata['AcquisitionDateTime'].values))
            mdata['TimeDifferenceSeconds'] = mdata['TimeDifference'].astype('float') / 1e9
            mdata['cumulative_time_secs'] = np.cumsum(mdata['TimeDifferenceSeconds'].values)
            mdata['ID'] = ID

            exposures = mdata['ExposureinmAs'].astype(np.float32).values
            mdata.to_pickle(os.path.join(pid, ID + '_metadata.pic'))
            mdata.to_excel(os.path.join(pid, ID + '_metadata.xlsx'))

        p_ctp += '.nii.gz'
        if not os.path.exists(p_ctp):
            continue
        to_process = [(p_ctp, ctp, pid_nii, '')]
        # Use frame 1 to generate a brainmask --> for CTA generation
        p_mask = os.path.join(pid, f'{ID}_multimask.nii.gz')
        if mask_roi_subset is not None and not os.path.exists(p_mask):
            if ctp is None:
                ctp = sitk.ReadImage(p_ctp)
            frame0 = ctp[:, :, :, 0]
            p_frame0 = os.path.join(pid, 'frame0.nii.gz')
            sitk.WriteImage(frame0, p_frame0)

            totalsegmentator(p_frame0, p_mask,
                             ml=True, fast=False,
                             roi_subset=mask_roi_subset, device='gpu',
                             verbose=False, nr_thr_saving=6, nr_thr_resamp=6)
            os.remove(p_frame0)

        p_ctp_reg = os.path.join(pid, filename + 'reg.nii.gz')
        if not os.path.exists(p_ctp_reg) and register:
            if ctp is None:
                # register ctp can also use path of ctp to load
                ctp_reg = register_ctp_frames(p_ctp, rp, p_ctp_reg)
            else:
                ctp_reg = register_ctp_frames(ctp, rp, p_ctp_reg)
        if register:
            to_process.append((p_ctp_reg, ctp_reg, pid_reg, '_reg'))
        # run the same analyses for ctp and ctp_reg

        # to_process defined above, can include original and registered images
        for (path, scan, pid_sav, addname) in to_process:

            if scan is None:
                if os.path.exists(path):
                    scan = sitk.ReadImage(path)
                else:
                    continue

            if mdata is None:
                mdata = pd.read_pickle(os.path.join(pid, ID + '_metadata.pic'))
                exposures = mdata['ExposureinmAs'].astype(np.float32).values

            # compute average and percentiles of the entire timeframe array
            if (isinstance(time_averages, list) or isinstance(time_averages, np.ndarray)):
                p_tavg = os.path.join(pid_sav, 'time_averages')
                __ = ctp_all_timeframes_averaging(ctp=scan,
                                                  exposures=exposures,
                                                  global_avg=time_averages,
                                                  sav_dir=p_tavg,
                                                  nnunet_folder=nnunet_folder,
                                                  ID=ID,
                                                  addname=addname,
                                                  overwrite=overwrite)

            if os.path.exists(p_mask) and not 'brain_mean_per_frame' + addname in mdata.columns:
                mean_per_frame = mean_brain_hu_per_frame(ctp=scan, bm=p_mask)
                # alter mdata and save
                mdata['brain_mean_per_frame' + addname] = mean_per_frame
                mdata.to_pickle(os.path.join(pid, ID + '_metadata.pic'))
                mdata.to_excel(os.path.join(pid, ID + '_metadata.xlsx'))
                cta_ix = np.argmax(mean_per_frame)
            elif 'brain_mean_per_frame' + addname in mdata.columns:
                cta_ix = np.argmax(mdata['brain_mean_per_frame' + addname])
            else:
                cta_ix = None

            if cta_ix is not None or all_ctas:
                __ = ctp2cta(scan,
                             exposures=exposures,
                             margin=margin,
                             ID=ID,
                             dir_sav=pid_sav,
                             nnunet_folder=nnunet_folder,
                             addname=addname,
                             cta_ixmax=cta_ix,
                             all_ctas=all_ctas,
                             overwrite=overwrite)





# legacy version used for sorens initial data
# def process_ctp_frames_separate(mdata,
#                                param_file=None,
#                                sloc_pid = None,
#                                clip_bounds=None,
#                                register=True
#                                ):
#
#     if sloc_pid is not None:
#         sloc_org_nii = os.path.join(sloc_pid, 'NII')
#         sloc_reg_nii = os.path.join(sloc_pid, 'REG')
#         sloc_trans_files = os.path.join(sloc_pid, 'transform_params')
#     else:
#         sloc_org_nii = None
#         sloc_reg_nii = None
#         sloc_trans_files = None
#
#     minVolno = mdata['Volno'].min()
#     out4D = []
#     out4D_reg = []
#     for no, file in tqdm(zip(list(mdata['Volno']), list(mdata['dcmfile']))):
#         # print(no,file, type(file), os.path.exists(file))
#         img = sitk.ReadImage(file)
#         img = sitk.Cast(sitk.Clamp(img, lowerBound=-1024, upperBound=1000), sitk.sitkInt16)
#         out4D.append(img)
#         if sloc_org_nii is not None:
#             exists(sloc_org_nii)
#             sitk.WriteImage(img, os.path.join(sloc_org_nii, str(no) + '.nii.gz'))
#         if register:
#             # store first volume to register to
#             if no == minVolno:
#                 start_img = img
#                 out4D_reg.append(img)
#                 if sloc_reg_nii is not None:
#                     exists(sloc_reg_nii)
#                     sitk.WriteImage(start_img, os.path.join(sloc_reg_nii, str(no) + '.nii.gz'))
#             else:
#                 reg_img, transforms = itk_register(start_img, img, param_file, fix_clip=clip_bounds, moving_clip=clip_bounds)
#                 reg_img = sitk.Cast(reg_img, sitk.sitkInt16)
#                 out4D_reg.append(reg_img)
#                 # store registered file
#                 if sloc_reg_nii is not None:
#                     exists(sloc_reg_nii)
#                     sitk.WriteImage(reg_img, os.path.join(sloc_reg_nii, str(no) + '.nii.gz'))
#                 # store reg params
#                 if sloc_trans_files is not None:
#                     exists(sloc_trans_files)
#                     transform_file = os.path.join(sloc_trans_files, 'transform_{}.txt'.format(no))
#                     itk.ParameterObject.New().WriteParameterFile(transforms, transform_file)
#
#     str_datetime = str(datetime2str(mdata['AcquisitionDateTime'].to_list()))
#     str_t_diff = str(timedelta2str(mdata['TimeDifference'].to_list()))
#
#     #add metadata to 4D sequence
#     if len(out4D)>2:
#         out4D = sitk.JoinSeries(out4D)
#         out4D.SetMetaData('AcquisitionDateTime', str_datetime)
#         out4D.SetMetaData('TimeDifference', str_t_diff)
#         out4D = sitk_add_tags(out4D, mdata, tags=['ExposureinmAs', 'KVP', 'CTDIvol'])
#         if sloc_pid is not None:
#             sitk.WriteImage(out4D, os.path.join(sloc_pid,'CTP.nii.gz'))
#
#     if len(out4D_reg)>2:
#         out4D_reg = sitk.JoinSeries(out4D_reg)
#         out4D_reg.SetMetaData('AcquisitionDateTime', str_datetime)
#         out4D_reg.SetMetaData('TimeDifference', str_t_diff)
#         out4D_reg = sitk_add_tags(out4D_reg, mdata, tags=['ExposureinmAs', 'KVP', 'CTDIvol'])
#         if sloc_pid is not None:
#             sitk.WriteImage(out4D_reg, os.path.join(sloc_pid,'CTP_reg.nii.gz'))
#
#     return out4D, out4D_reg

