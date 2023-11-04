import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm, transforms
import seaborn as sns


def plot_volume(img, pct_lst=None, titl=''):
	if pct_lst is None:
		pct_lst = [.1, .2, .4, .6]
	f,axs = plt.subplots(1,len(pct_lst), figsize=(20,5)) # y,x
	z = img.shape[0]
	for pct,ax in zip(pct_lst,axs):
		ax.imshow(img[int(z*pct),:,:], cmap='gray')
		ax.axis('off')
	f.suptitle(titl, fontsize=16)
	plt.show()
	return f
	
def sidebyside_plot_volume(imgs, pct_lst=None,
						   titl='', sloc=None):

	if pct_lst is None:
		pct_lst = [.2, .6]

	f,axs = plt.subplots(1,len(pct_lst)*len(imgs)+1, figsize=(20,5)) # y,x
	ix=0
	# use one box for the title
	ax = axs[ix]
	ax.imshow(np.full_like(imgs[0][0,:,:],imgs[0].min()), cmap='gray')
	ax.annotate(titl, color='w', fontsize=20, xy=(.5,.5), 
				xycoords=ax.transAxes, ha='center')
	ax.axis('off')
	for pct in pct_lst:
		for img in imgs:
			z = img.shape[0]
			ax = axs[ix+1]
			ax.imshow(img[int(z*pct),:,:], cmap='gray')
			ax.axis('off')
			ix+=1
	plt.subplots_adjust(wspace=0)
	return f

def sidebyside_plot_volume_multidim(imgs, axial=None,
									saggital=None, coronal=None, titl='',
									addtitl = '\nCTA-NCCT', sloc=None):
	
	if axial is None:
		axial = [.2, .6]
	n_subplots = 1
	
	if axial!=None:
		n_subplots += len(axial)*len(imgs)
	if saggital!=None:
		n_subplots += len(saggital)*len(imgs)
	if coronal!=None:
		n_subplots += len(coronal)*len(imgs)
		
	f,axs = plt.subplots(1,n_subplots, figsize=(20,5)) # y,x
	ix=0
	# use one box for the title
	ax = axs[ix]
	ax.imshow(np.full_like(imgs[0][0,:,:],imgs[0].min()), cmap='gray')
	ax.annotate(titl+addtitl, color='w', fontsize=20, xy=(.5,.5), 
				xycoords=ax.transAxes, ha='center')
	ax.axis('off')
	if axial!=None:
		for axia in axial:
			for img in imgs:
				dim = img.shape[0]
				ax = axs[ix+1]
				ax.imshow(img[int(dim*axia),:,:], cmap='gray')
				ax.axis('off')
				ix+=1
			
	if saggital!=None:
		for sag in saggital:
			for img in imgs:
				dim = img.shape[1]
				ax = axs[ix+1]
				ax.imshow(img[:,int(dim*sag),:], cmap='gray')
				ax.axis('off')
				ix+=1
				
	if coronal!=None:
		for cor in coronal:
			for img in imgs:
				dim = img.shape[2]
				ax = axs[ix+1]
				ax.imshow(img[:,:,int(dim*cor)], cmap='gray')
				ax.axis('off')
				ix+=1       
	plt.subplots_adjust(wspace=0)
	if sloc!=None:
		f.savefig(os.path.join(sloc,titl+'.png'))
	return f

def get_plt_cm(img,cmap='gray'):
	my_cm = cm.get_cmap(cmap)
	normed_img = (img - np.min(img)) / (np.max(img) - np.min(img))
	return my_cm(normed_img)*255

def bland_altman_plot(data1, data2, *args, **kwargs):
	# see: https://stackoverflow.com/questions/16399279/bland-altman-plot-in-python
	data1 = np.asarray(data1)
	data2 = np.asarray(data2)
	mean = np.mean([data1, data2], axis=0)
	diff = data1 - data2  # Difference between data1 and data2
	md = np.mean(diff)  # Mean of the difference
	sd = np.std(diff, axis=0)  # Standard deviation of the difference

	agreement = 1.96
	high, low = md + 1.96 * sd, md - 1.96 * sd

	ax = sns.scatterplot(mean, diff, *args, **kwargs)
	ax.set_xlim(0, mean.max() + int(mean.max() * .2))
	ax.set_ylim(min(diff.min() * 1.1, low * 1.6), max(diff.max() * 1.1, high * 1.6))
	# print(diff.min()*1.1,low*1.25,diff.max()*1.1,high*1.25)
	ax.axhline(md, color='black', linestyle='-')
	ax.axhline(high, color='black', linestyle='--')
	ax.axhline(low, color='black', linestyle='--')

	loa_range = high - low
	offset = (loa_range / 100.0) * 1.5
	trans = transforms.blended_transform_factory(
		ax.transAxes, ax.transData)
	xloc = 0.98
	plt.text(xloc, md + offset, 'Mean', ha="right", va="bottom",
			 transform=trans)
	plt.text(xloc, md - offset, '%.0f' % round(md), ha="right",
			 va="top", transform=trans)
	plt.text(xloc, high + offset, '+%.0f SD' % agreement, ha="right",
			 va="bottom", transform=trans)
	plt.text(xloc, high - offset, '%.0f' % high, ha="right", va="top",
			 transform=trans)
	plt.text(xloc, low - offset, '-%.0f SD' % agreement, ha="right",
			 va="top", transform=trans)
	plt.text(xloc, low + offset, '%.0f' % low, ha="right", va="bottom",
			 transform=trans)