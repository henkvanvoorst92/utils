import numpy as np
import os
import argparse
import ast
import json
from collections import defaultdict
import re
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import SimpleITK as sitk
import logging
import torch
import zipfile

def exists(p,create=True):
    if not os.path.exists(p):
        if create:
            os.makedirs(p)


def args2dct(args):
    dct = vars(args)
    for k,v in dct.items():
        if isinstance(v,type):
            dct[k] = str(v)
        elif isinstance(v,np.ndarray):
            dct[k] = str(v)
    return dct

def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data
def store_opt_json(opt):
    p_json = os.path.join(opt.loc_checkpoints,'opt.json')
    if not os.path.exists(opt.loc_checkpoints):
        os.makedirs(opt.loc_checkpoints)
    dct = args2dct(opt)
    with open(p_json, 'w', encoding='utf-8') as f:
        json.dump(dct, f, ensure_ascii=False, indent=4)

def load_opt_json(root, org_opt=None):
    p_json = os.path.join(root,'opt.json')
    if not os.path.exists(p_json):
        p_json = os.sep+p_json
    with open(p_json) as f:
        dct = json.load(f)
    for k,v in dct.items():
        if k=='norm':
            dct[k] = getattr(torch.nn,v.strip("<>''").split('.')[-1])

    dct['loc_checkpoints'] = root
    if org_opt is not None:
        for argname in vars(org_opt):
            if argname not in dct.keys():
                dct[argname] = getattr(org_opt,argname)
    return argparse.Namespace(**dct)

def convert_str_args(args,args_to_convert):
    dct = {}
    for arg in vars(args):
        if arg in args_to_convert:
            var = getattr(args,arg)
            if isinstance(var,str):
                var = ast.literal_eval(var)
            dct[arg] = var
        else:
            dct[arg] = getattr(args,arg)
    return argparse.Namespace(**dct)

def is_odd(num):
    return num % 2 != 0

def make_odd(num):
    return np.ceil((num + 1)/2)*2 - 1

def all_tag_values(dcm, mdct=None, stringconvert=False):
    # returns a set of dicom tags stripped from dcm (a dicom file)
    # mdct is a dictionary with
    # keys = dicom tag number (example: (0x0020, 0x0032))
    # values = corresponding dicom tag name ('ImagePositionPatient')
    # stringconvert is optional and converts all tags to strings
    out = []
    if mdct is None:
        mdct = create_mdct(dcm)
    for k, v in mdct.items():
        tag = get_tag_value(dcm, k, v)
        if stringconvert:
            tag = str(tag)
        out.append(tag)
    return out

def create_mdct(dcm):
    # extracts all metadata names and IDs
    mdct = {}
    # iterating over items does not work for v
    for k in dcm.keys():
        try:
            n = re.subn('[ ]()', '', str(dcm[k].name))[0]
            if n == 'PixelData':
                continue
            mdct[n] = k
        except:
            print('Does not work:', dcm[k])
    return mdct

def get_tag_value(dcm, tagno, tagname):
    """
    Returns the dicom tag value based on e a string name (tagname)
    or number (tagno: (0x0001,0x0001))
    """
    try:
        out = dcm[tagno].value
    except:
        try:
            out = dcm[tagname].value
        except:
            out = np.NaN
    return out


## function to extract all dicom tags
def get_tagnos_tagnames(dcm):
    """
    Returns lists of tag numbers and names from dicom
    tagno example: (0x0001,0x0001)
    """
    tagnos = list(dcm.keys())
    name2no = {}
    no2name = {}
    for tagno in tagnos:
        tagname = str(dcm[tagno].name)
        tagname = tagname.replace(' ', '')
        name2no[tagname] = tagno
        no2name[tagno] = tagname
    return name2no, no2name


def try_t_str(datestring:str, t_str:list):
    """
    Try different string formats provided in t_str (list)
    for a given datestring, returns the most appropriate found
    Warning!: the order of t_str is important, if a str in t_str
    fits the datastring the remaining t_str is not considered
    """
    if t_str is None:
        t_str= ["%Y%m%d%H%M%S", "%Y%m%d%H%M%S.%f",  # datetimeformats
                   "%d%m%Y%H%M%S", "%d%m%Y%H%M%S.%f",  # datetimeformats
                   "%m%d%Y%H%M%S.%f" "%m%d%Y%H%M%S",  # datetimeformats
                   "%Y%m%d", "%d%m%Y", "%m%d%Y",  # date formats
                   "%H%M%S", "%H%M%S.%f",  # time formats
                   ]



    for ts in t_str:
        try:
            out = datetime.strptime(datestring, ts)
            break
        except:
            out = np.NaN
            continue
    return out


def is_ascending(numbers):
    """
    Chatgpt 4
    Check if the given NumPy array of numbers is in ascending order using vectorized operations.

    Parameters:
    numbers (np.array): A NumPy array of numbers.

    Returns:
    bool: True if the array is in ascending order, False otherwise.
    """
    # Convert list to a NumPy array if it's not already
    if not isinstance(numbers, np.ndarray):
        numbers = np.array(numbers)

    # Perform a vectorized comparison between shifted versions of the array
    return np.all(numbers[:-1] <= numbers[1:])

def list_files(startpath):
    #source: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def compute_time_differences(datetime_list):
    # Use list comprehension to compute the difference between each datetime and the previous one
    time_differences = [datetime_list[i] - datetime_list[i - 1] for i in range(1, len(datetime_list))]

    time_differences.insert(0, 0)  # or timedelta(0) if you prefer

    return time_differences


def datetime2str(datetimes: List[datetime],
                 date_format: str = "%Y%m%d%H%M%S.%f") -> List[str]:
    """
    Convert a list of datetime objects to a list of strings.

    :param datetimes: List of datetime.datetime objects.
    :param date_format: The format in which to convert the datetimes to strings.
    :return: List of datetime strings in the given format.
    """
    return [dt.strftime(date_format) if dt is not None else '0' for dt in datetimes]


def timedelta2str(timedeltas: List[timedelta]) -> List[str]:
    """
    Convert a list of timedelta objects to a list of strings.
    Each timedelta is represented by total seconds and microseconds.

    :param timedeltas: List of datetime.timedelta objects.
    :return: List of strings with each timedelta represented by "total_seconds.microseconds".
    """
    out = []
    for td in timedeltas:
        if td is None or isinstance(td, pd._libs.tslibs.nattype.NaTType):  # or np.isnan(td)
            out.append('0')
        else:
            out.append('{}'.format(td.total_seconds()))

    return out

def np2sitk(arr: np.ndarray, original_img: sitk.SimpleITK.Image):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(original_img.GetSpacing())
    img.SetOrigin(original_img.GetOrigin())
    img.SetDirection(original_img.GetDirection())
    # this does not allow cropping (such as removing thorax, neck)
    #img.CopyInformation(original_img)
    return img


def initialize_logging(log_file_path):
    # Configure the basic settings for global logging in one file (allways keeps runin in kernel)
    logging.basicConfig(filename=log_file_path,
                        level=logging.CRITICAL,  # Capture all levels of logging messages
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        # Include timestamp, logger name, log level, and log message
                        datefmt='%Y-%m-%d %H:%M:%S')  # Format for the timestamp
# print(f"Logging initialized. Log messages will be written to {log_file_path}")
def initialize_logger_handler(log_file_path,
                              remove_existing=False,
                              level=logging.DEBUG):
    if remove_existing and os.path.exists(log_file_path):
        os.remove(log_file_path)

    # Configure the logger
    logger = logging.getLogger(log_file_path)  # Get custom logger
    logger.setLevel(level)

    # Create a file handler that logs messages to the specified file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)

    # Create a formatter and set it for the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger, file_handler


def stop_logging(logger, file_handler):
    logger.removeHandler(file_handler)  # Remove the specified handler
    file_handler.close()  # Close the handler to free up resources


def any_in_list_has_str(lst, string, max_txt_length=None):
    #checks if any of a list of strings has a certain string
    if max_txt_length is not None:
        return np.any([string in txt for txt in lst if len(txt) < max_txt_length])
    else:
        return np.any([string in txt for txt in lst])


def string_in_list(lst, string, max_txt_length=None):
    #checks if a specific string is in a list of strings
    if max_txt_length is not None:
        return [string in txt for txt in lst if len(txt) < max_txt_length]
    else:
        return [string in txt for txt in lst]


# do this
def multistring_in_list(lst, string, max_txt_length=None):
    # checks if strings in a list of strings (string)
    #occur in any of the strings in lst
    if max_txt_length is not None:
        return [np.any([st in txt for st in string]) for txt in lst if len(txt) < max_txt_length]
    else:
        return [np.any([st in txt for st in string]) for txt in lst]

def match_lists_of_strings(arr, scan, location, exclusion=None):
    """
    Cheks if any from scan and location lists are occuring in
    any of the strings in arr
    """

    # Pre-compile lists into sets for faster lookup
    scan_set = set([s.lower() for s in scan])
    location_set = set([l.lower() for l in location])
    if exclusion is not None:
        excl_set = set([ex.lower() for ex in exclusion])

    # Function to check if any item in a set is a substring of the text
    def contains_any(text, string_set):
        return any(s in text for s in string_set)

    if exclusion is not None:
        result = [contains_any(text.lower(), scan_set) and \
                  contains_any(text.lower(), location_set) and not \
                      contains_any(text.lower(), excl_set) for text in arr]
    else:
        # Check each text in arr for any match in both scan and location
        result = [contains_any(text.lower(), scan_set) and \
                  contains_any(text.lower(), location_set) for text in arr]

    return result
def find_files_with_ID(ID,path):
    file_w_ID = [os.path.join(path,file) for file in os.listdir(path) if ID in file ]
    if len(file_w_ID)==1:
        file_w_ID = file_w_ID[0]
    return file_w_ID

def download_pd_with_ext(p, ext='.csv', dtype=None):
    # Mapping of file extensions to their respective pandas read function
    pd_readers = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.json': pd.read_json,
        '.html': pd.read_html,  # This will return a list of DataFrames
        '.sql': pd.read_sql,  # This requires a connection argument, handle separately
        '.parquet': pd.read_parquet,
        '.feather': pd.read_feather,
        '.ftr': pd.read_feather,
        '.dta': pd.read_stata,
        '.pkl': pd.read_pickle,
        '.pic': pd.read_pickle,
        # Add more formats as needed
    }

    if ext not in pd_readers:
        raise ValueError(f"Unsupported file extension: {ext}")

    pd_reader = pd_readers[ext]
    dct = {}
    for f in os.listdir(p):
        if not f.endswith(ext):
            continue
        file_path = os.path.join(p, f)
        # Special handling for HTML and SQL as they might require additional processing
        if ext == '.html':
            dct[f] = pd_reader(file_path)[0]  # Assuming you want the first table
        elif ext == '.sql':
            # Example: Assume `conn` is your database connection
            # You would replace `conn` with your actual connection object
            # dct[f] = pd_reader('your_query_here', conn)
            pass  # Implement SQL reading based on your database connection
        elif ext in ['.pic','.pkl', '.feather', '.ftr']:
            dct[f] = pd_reader(file_path)
        else:
            dct[f] = pd_reader(file_path, dtype=dtype)

    return dct

def write_dfs_with_new_ext(dfs_dict, path_to_storage, new_exts):
    """
    Iterates over a dictionary of filenames and DataFrames, changes the filenames' extensions,
    and writes the files to a specified storage path in all the provided new formats.

    Parameters:
    - dfs_dict: Dictionary where keys are filenames and values are pandas DataFrames.
    - path_to_storage: The file path where the new files will be stored.
    - new_exts: List of new file extensions to store the DataFrames.
    """

    # Mapping of file extensions to their respective pandas write function
    pd_writers = {
        '.csv': 'to_csv',
        '.xlsx': 'to_excel',
        '.json': 'to_json',
        '.html': 'to_html',
        '.parquet': 'to_parquet',
        '.feather': 'to_feather',
        '.ftr': 'to_feather',
        '.dta': 'to_stata',
        '.pkl': 'to_pickle',
        '.pic': 'to_pickle',
        # Add more formats as needed
    }
    
    for filename, df in dfs_dict.items():
        # Strip the original extension and prepare the base filename
        base_filename = os.path.splitext(filename)[0]

        for ext in new_exts:
            if ext in pd_writers:
                # Construct the new filename with the desired extension
                new_filename = f"{base_filename}{ext}"
                # Construct the full path to the new file
                full_path = os.path.join(path_to_storage, new_filename)

                # Get the corresponding pandas function and call it on the DataFrame
                getattr(df, pd_writers[ext])(full_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

def zip_directory_tree(p_zip):
    with zipfile.ZipFile(p_zip, 'r') as zip_ref:
        # Create a dictionary to hold the directory structure
        dir_tree = defaultdict(list)
        # Populate the directory tree with files
        for file_info in zip_ref.infolist():
            directory = os.sep.join(file_info.filename.split(os.sep)[:-1])
            dir_tree[directory].append(file_info.filename)

    return dir_tree

def rename_duplicate_IDs(ids):
    count_dict = {}  # To track the counts of each ID
    renamed_ids = []  # To store the renamed IDs
    for id in ids:
        if id in count_dict:
            # Increment the count for this ID
            count_dict[id] += 1
            # Append a letter to the ID based on its occurrence count
            renamed_ids.append(f"{id}-{chr(96 + count_dict[id])}")
        else:
            # Add the ID to the dictionary with a count of 1
            count_dict[id] = 1
            # Keep the first occurrence of the ID unchanged
            renamed_ids.append(id)
    return renamed_ids

def is_notebook():
    """Check if the script is running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def int_convert_list_args(args_list):
    # Try to convert to integer, fallback to string if it fails
    out = []
    for item in args_list:
        try:
            processed_item = int(item)
        except ValueError:
            processed_item = item
        out.append(processed_item)

    return out

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def write_list2txt(path: str, data_list: list):
    with open(path, 'w') as file:
        # Write each item on a new line
        for item in data_list:
            file.write(str(item) + '\n')


def read_list_from_txt(path):
    retrieved_list = []
    # Open the text file for reading
    with open(path, 'r') as file:
        # Read each line from the file
        for line in file:
            # Strip the newline character and add to the list
            retrieved_list.append(ast.literal_eval(line.strip()))
    return retrieved_list
