import torch

def _safe_flatten(x):
	return x.contiguous().view(x.size(0), -1)

def dice_score(preds, targets, eps=1e-6):
	#Compute Dice score per-batch and return average.
	preds_f = _safe_flatten(preds)
	targets_f = _safe_flatten(targets)

	intersection = (preds_f * targets_f).sum(dim=1)
	denom = preds_f.sum(dim=1) + targets_f.sum(dim=1)
	dice = (2.0 * intersection + eps) / (denom + eps)
	return dice.mean().item()


def iou_score(preds, targets, eps=1e-6):
	preds_f = _safe_flatten(preds)
	targets_f = _safe_flatten(targets)
	intersection = (preds_f * targets_f).sum(dim=1)
	union = preds_f.sum(dim=1) + targets_f.sum(dim=1) - intersection
	iou = (intersection + eps) / (union + eps)
	return iou.mean().item()
