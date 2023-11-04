import numpy as np
from numba import njit
from sklearn import metrics
from utils.utils import rtrn_np

def np_dice(y_true,y_pred,add=1e-6):
	return (2*(y_true*y_pred).sum()+add)/(y_true.sum()+y_pred.sum()+add)

def Get_dice(gt, dmap, thresholds, masks=None, largerthan=True):
	if masks is None:
		masks = []
	out = []
	for t in thresholds:
		if largerthan:
			seg = (dmap>t)*1
		else:
			seg = (dmap<t)*1
		
		dice1 = round(np_dice(gt,seg),6)
		row = [t,dice1]
		
		for m in masks:
			segm = seg*m
			dicem = round(np_dice(gt,segm),6)
			row.append(dicem)
		out.append(row)
	return out

@njit(parallel=True,fastmath=True)
def Fast_dice(gt,dmap,thresholds, masks=None, largerthan=True):
	if masks is None:
		masks = []
	out = []
	for t in thresholds:
		if largerthan:
			seg = (dmap>t)*1
		else:
			seg = (dmap<t)*1
		
		dice1 = (2*(gt*seg).sum())/(gt.sum()+seg.sum())
		row = [t,dice1]
		
		#if isinstance(mask,np.ndarray):
		for m in masks:
			segm = seg*m
			dicem = (2*(gt*segm).sum())/(gt.sum()+segm.sum())
			row.append(dicem)
		out.append(row)
	return out

def torch_dice(gt, segs, thresholds, masks=[], device='cpu', add_data=None):
	if add_data is None:
		add_data = []
	gt = gt.to(device)
	segs = [seg.to(device) for seg in segs]
	masks = [m.to(device) for m in masks]
	
	out = []
	for t in thresholds:
		for seg in segs:
			dice1 = (2*(gt*seg).sum())/(gt.sum()+seg.sum())
			row = [*add_data,t,dice1.item()]
			#if isinstance(mask,np.ndarray):
			for m in masks:
				segm = np.copy(seg)*m
				gtm= np.copy(gt)*m
				dicem = (2*(gtm*segm).sum())/(gtm.sum()+segm.sum())
				row.append(dicem.item())
		out.append(row)
	return out

def torch_multiseg_dice(gt,dmap,thresholds,SegmentationAlternatives=None, masks=[], device='cpu', add_data=[], return_segs=False):
	if not isinstance(gt,np.ndarray):
		gt = rtrn_np((gt>0)).astype(np.int8)
	if not isinstance(dmap,np.ndarray):
		dmap = rtrn_np(dmap)
	if len(masks)>0: #masks[0] should be baseline brainmask
		masks = [rtrn_np((m>0)).astype(np.int8) if not isinstance(m,np.ndarray) else m for m in masks]
		gt = gt*masks[0] #adjust ground truth to mask
		dmap = dmap*masks[0]
	#gt = gt.to(device)
	#dmaps = [dmap.to(device) for dmap in dmaps]
	#dmap = dmap.to(device)
	#masks = [m.to(device) for m in masks]
	out = []
	out_segs = []
	best_dice = 0
	for t in thresholds:
		seg = (dmap>t).astype(np.int8) # use mask 1 here (no more computation without masks)
		if SegmentationAlternatives is None:
			segs = [seg]
			names = ['seg']
		else:
			segs,names = SegmentationAlternatives(seg) #run seg as torch tensor

		for seg,segname in zip(segs,names):
			dice1 = (((gt*seg).sum())*2)/(gt.sum()+seg.sum())
			#if isinstance(mask,np.ndarray):
			row = [*add_data,t,segname,'org_bm',dice1] #.item()
			out.append(row)
			if dice1>best_dice and return_segs:
				best_dice = dice1
				out_segs = segs
			
			if len(masks)>1:
				for maskname,m in enumerate(masks[1:]):
					segm = seg*m
					dicem = (((gt*segm).sum())*2)/(gt.sum()+segm.sum())
					row = [*add_data,t,segname,'mask_'+str(maskname),dicem] #.item()
					out.append(row)
					if dicem>best_dice and return_segs:
						best_dice = dicem
						out_segs = [*segs,m]

	if return_segs:
		return out,out_segs
	else:
		return out


def metrics_outcome(y_true, score, thres=0.5):  # score=y_pred
	y_pred = (score > thres) * 1

	tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

	accuracy = round(((tp + tn) / (tn + fp + fn + tp)) * 100, 2)
	recall = round(((tp) / (fn + tp)) * 100, 2)  # =TPR
	specificity = round(tn / (tn + fp) * 100, 2)  # =TNR
	precision = round(((tp) / (tp + fp)) * 100, 2)  # =PPV
	NPV = round(((tn) / (tn + fn)) * 100, 2)

	fpr, tpr, thresholds = metrics.roc_curve(y_true, score)
	roc_auc = round(metrics.roc_auc_score(y_true, y_pred) * 100, 2)

	out = {'acc': accuracy,
		   'TPR': recall, 'TNR': specificity,
		   'PPV': precision, 'NPV': NPV,
		   'AUC': roc_auc,
		   'y_pred=1_%': round(y_pred.sum() / y_pred.shape[0] * 100, 2),
		   'y_true=1_%': round(y_true.sum() / y_true.shape[0] * 100, 2),
		   'y_pred=1_abs': y_pred.sum(),
		   'y_true=1_abs': y_true.sum()}

	return out  # accuracy, recall, precision, specificity, roc_auc#, fpr, tpr, y_pred, y_prob

