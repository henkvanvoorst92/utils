import numpy as np
import os
import argparse
from torch import nn
import ast
import json
from datetime import datetime

def exists(p,create=True):
	if not os.path.exists(p):
		if create:
			os.makedirs(p)

def rtrn_np(img): # returns numpy array from torch tensor (on cuda)
	return img.detach().cpu().numpy()

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
			dct[k] = getattr(nn,v.strip("<>''").split('.')[-1])

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

def all_tag_values(dcm,mdct, stringconvert=False):
	# returns a set of dicom tags stripped from dcm (a dicom file)
	# mdct is a dictionary with
		#keys = dicom tag number (example: (0x0020, 0x0032))
		# values = corresponding dicom tag name ('ImagePositionPatient')
	# stringconvert is optional and converts all tags to strings
	out = []
	for k,v in mdct.items():
		tag = get_tag_value(dcm,k,v)
		if stringconvert:
			tag = str(tag)
		out.append(tag)
	return out

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