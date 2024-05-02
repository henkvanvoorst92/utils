
import numpy as np
import torch
from numba import njit
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

@njit(fastmath=True)
def get_3Dmask_coordinates(mask:np.ndarray,foreground=1):
    """
    returns a list of coordinates from a mask
    with foreground pixels
    """

    f_coordinates = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                #assert np.isscalar(mask[i, j, k])
                if mask[i,j,k]==foreground:
                    f_coordinates.append([i,j,k])
    return f_coordinates

@njit(fastmath=True)
def get_2Dmask_coordinates(mask:np.ndarray,foreground=1):
    """
    returns a list of coordinates from a mask
    with foreground pixels
    """

    f_coordinates = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j]==foreground:
                f_coordinates.append([i,j])

    return f_coordinates

@njit(parallel=True ,fastmath=True)
def get_mask_coordinates(mask :np.ndarray ,foreground=1, is3D=False):
    """
    returns a list of coordinates from a mask
    with foreground pixels
    """

    f_coordinates = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not is3D:
                if mask[i ,j ] == foreground:
                    f_coordinates.append([i ,j])
            else:
                for k in range(mask.shape[2]):
                    if mask[i ,j ,k ] == foreground:
                        f_coordinates.append([i ,j ,k])
    return f_coordinates


# @njit(parallel=True ,fastmath=True) #does not work
def coordinates2mask_2D(coords, mask, C=1):
    # coords should be np array or list of coordinate points
    # all coordinates are set in the mask
    # values are all c (if int) or
    if isinstance(C, int):
        C = np.ones(len(coords))
    elif (isinstance(C, np.ndarray) or isinstance(C, list)) and len(coords) == len(C):
        C = C
    else:
        raise Exception('c is of wrong type, only np.array, list, or int allowed:', C)

    for c, (x, y) in zip(C, coords):
        mask[x, y] = c

    return mask


# @njit(parallel=True ,fastmath=True) #does not work
def coordinates2mask_3D(coords, mask, C=1):
    # coords should be np array or list of coordinate points
    # all coordinates are set in the mask
    # values are all c (if int) or

    if isinstance(C, int):
        C = np.ones(len(coords))
    elif (isinstance(C, np.ndarray) or isinstance(C, list)) and len(coords) == len(C):
        C = C
    else:
        raise Exception('c is of wrong type, only np.array, list, or int allowed:', C)

    for c, (x, y, z) in zip(C, coords):
        mask[x, y, z] = c

    return mask


# this functions but is not really fast
@njit(fastmath=True)
def parallel_coordinates2mask_3D(coords, mask, C):
    # coords should be np array or list of coordinate points
    # all coordinates are set in the mask
    # C represents the values to fil in
    for ix in range(len(coords)):
        mask[coords[ix]] = C[ix]
    return mask


def euclidean_distance(coords1, coords2, spacing=None, floattype=np.float32):
	# computes the euclidean distance between all coords1 and coords2
	# spacing (dims=x,y,z) per dimension can be considered for distance computation
	# returns a len(coords1)*len(coords2) array

	diffs = (coords1[:, None, :] - coords2)
	if spacing is not None:
		# control type if spacing used otherwise computation overload
		diffs = diffs.astype(floattype) * np.array(spacing).astype(floattype)
	# compute euclidean distance for each segmentation point
	# distances dims= seg_coordinate,centerline_coordinate
	# einsum: https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
	distances = np.sqrt(np.einsum('ijk,ijk->ij', diffs, diffs))
	return distances


def torch_euclidean_distance(coords1, coords2, spacing=None, floattype=torch.float32):
	# torch version of euclidean distance function
	# computes the euclidean distance between all coords1 and coords2
	# spacing (dims=x,y,z) (torch tensor on same dev) per dimension can be considered for distance computation
	# returns a len(coords1)*len(coords2) array

	diffs = (coords1[:, None] - coords2).type(floattype)
	if spacing is not None:
		# control type if spacing used otherwise computation overload
		diffs = diffs * spacing
	# compute euclidean distance for each segmentation point
	# distances dims= seg_coordinate,centerline_coordinate
	# einsum: https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
	distances = torch.sqrt(torch.einsum('ijk,ijk->ij', diffs, diffs))
	return distances

def batch_euclidean_distance_between(coords1, coords2,
									 max_chunk_size=10e3,
									 spacing=None,
									 floattype=np.float32,
									 arraytype='numpy',
									 dev='cpu',
									 direct_to_dev='',
									 return_ixs=False,
									 verbal=False):
	# coords1: coordinates of first mask
	# coords2: coordinate
	# max_chunk_size
	# spacing: tuple of 3 spacing in mm to adjust distance measures
	# dev: device used if arraytype if torch
	# direct_to_dev: all', 'coords1', or 'coords2' are directly allocated to device -->
	# this can save memory but might take time (coords1 only to device when batch is create)
	# returns lowest euclidean distance for each coord1 to any coords2

	if spacing is None:
		spacing = np.array([1, 1, 1])
	# xsp,ysp,zsp = spacing
	# spacing = np.array([xsp,ysp,zsp])

	if arraytype == 'torch':
		spacing = torch.tensor(spacing, requires_grad=False).type(floattype).to(dev)
		coords1 = torch.tensor(coords1, requires_grad=False)
		coords2 = torch.tensor(coords2, requires_grad=False)
		if direct_to_dev == 'all':
			coords1 = coords1.to(dev)
			coords2 = coords2.to(dev)
		if direct_to_dev == 'coords1':
			coords1 = coords1.to(dev)
		if direct_to_dev == 'coords2':
			coords2 = coords2.to(dev)

	elif arraytype == 'numpy':
		coords1 = np.array(coords1)
		coords2 = np.array(coords2)
		spacing = np.array(spacing).astype(floattype)

	if len(coords1) > max_chunk_size:
		rnge = np.arange(0, len(coords1) + max_chunk_size, max_chunk_size).astype(int)
		out = []
		iterable = range(0, len(rnge))
		if verbal:
			iterable = tqdm(iterable)
		for ix in iterable:
			if ix >= (len(rnge) - 1):
				continue
			batch_coords1 = coords1[rnge[ix]:rnge[ix + 1]]

			if arraytype == 'torch':
				if direct_to_dev == 'coords2':
					batch_coords1 = batch_coords1.to(dev)
				# compute euclidean distance with einsum
				distances = torch_euclidean_distance(batch_coords1, coords2, spacing=spacing)
				# distances dims are (batch_coords1_ix, coords2_ix)
				# compute index of centerline with shortes distance to chunk coords
				if return_ixs:
					# shortest_ixs: shortest distance from all coords 1 in batch to coords2
					data = distances.min(axis=1)
				else:
					data, __ = distances.min(axis=1)
			else:
				distances = euclidean_distance(batch_coords1, coords2, spacing=spacing)
				# compute index of centerline with shortes distance to chunk coords
				if return_ixs:
					data = np.min(distances, axis=1), np.argmin(distances, axis=1)
			out.append(data)

	if return_ixs:
		if arraytype == 'torch':
			distances = torch.hstack([o[0] for o in out])
			shortest_ix = torch.hstack([o[1] for o in out])
		elif arraytype == 'numpy':
			distances = np.hstack([o[0] for o in out])
			shortest_ix = np.hstack([o[1] for o in out])
		out = [distances, shortest_ix]
	else:
		if arraytype == 'torch':
			out = torch.hstack(out)
		elif arraytype == 'numpy':
			out = np.hstack(out)

	return out

def nearest_neighbor_distances(coords1, coords2, spacing=None):
	# Create a NearestNeighbors object
	if spacing is None:
		spacing = np.ones(3)
	else:
		spacing = np.array(spacing)

	# Scale the coordinates by the spacing
	coords1_scaled = coords1 * spacing
	coords2_scaled = coords2 * spacing

	# Create a NearestNeighbors object
	nn = NearestNeighbors(n_neighbors=1, algorithm='auto')  # 'auto' chooses the best algorithm
	nn.fit(coords2_scaled)  # Fit on scaled coords2

	# Find the nearest point in scaled coords2 for each point in scaled coords1
	distances, indices = nn.kneighbors(coords1_scaled)

	return distances.flatten()
