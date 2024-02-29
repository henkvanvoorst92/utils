import numpy as np
import os
import argparse
import ast
import json
import re
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import SimpleITK as sitk
import logging
import torch

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

