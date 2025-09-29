import torch
from torch.amp import autocast
import torch.nn.functional as F

def validate_model(model,test_loader,criterion,device,metrics_fn,scaler=None):
    model.eval()
    val_loss=0.0
    val_dice,val_iou=0.0,0.0

    with torch.no_grad():
        for images,masks in test_loader:
            images,masks=images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            if scaler is not None:
                with autocast('cuda'):
                    preds=model(images)
                    loss=criterion(preds,masks)
            else:
                preds=model(images)
                loss=criterion(preds,masks)
            
            dice,iou=metrics_fn(preds,masks)
            val_loss+=loss.item()
            val_dice+=dice
            val_iou+=iou
    avg_val_loss=val_loss/len(test_loader)
    avg_val_dice=val_dice/len(test_loader)
    avg_val_iou=val_iou/len(test_loader)

    return avg_val_loss, avg_val_dice, avg_val_iou

def calculate_metrics(preds, masks,dice_fn,iou_fn,threshold=0.5):
    preds_sigmoid = torch.sigmoid(preds)
    preds_binary = (preds_sigmoid > threshold).float()
    # ensure masks are binary
    masks_binary = (masks > 0.5).float()
    dice = dice_fn(preds_binary, masks_binary)
    iou = iou_fn(preds_binary, masks_binary)
    return dice, iou