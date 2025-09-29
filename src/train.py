import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from configs import seg_config as cfg
from src.dataset.brisc_dataset import BrainMRIDataset
from src.utils.metrics import dice_score, iou_score
from src.utils.augmentation import create_data_loaders
from src.utils.checkpoint import (save_checkpoint,ModelCheckpoint,resume_training)
from src.utils.early_stopping import EarlyStopping
from src.utils.io_utils import setup_directories
from src.utils.validation import (validate_model, calculate_metrics)
from src.model.unet import build_unet
from tqdm import tqdm

def train():

    #setting up directories and device
    setup_directories()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on device: {device}')

    #loading the dataset
    train_dataset=BrainMRIDataset(cfg.train_images, cfg.train_mask)
    test_dataset= BrainMRIDataset(cfg.test_images, cfg.test_mask)
    train_loader,test_loader=create_data_loaders(train_dataset,test_dataset, cfg)

    #loading the model with loss optimizer functions and sheduler
    model=build_unet(in_c= cfg.in_channel, out_c= cfg.out_channel).to(device)
    criterion=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler=CosineAnnealingLR(optimizer,T_max=cfg.epochs,eta_min=1e-6)

    #training utils
    use_amp=hasattr(torch.cuda,'amp') and torch.cuda.is_available()
    scaler=GradScaler('cuda') if use_amp else None
    early_stopping=EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    model_checkpoint=ModelCheckpoint('outputs/checkpoints/best_model_dice_{val_dice:.4f}_epoch_{epoch}.pth',monitor='val_dice',mode='max')

    #resume training if checkpoint exists or start fresh
    resume_path='outputs/checkpoints/last_checkpoint.pth'
    start_epoch,best_dice,train_history,val_history=resume_training(model, optimizer, scheduler, resume_path)
    print(f'Starting training from epoch {start_epoch+1}')

    #training loop
    for epoch in range(start_epoch,cfg.epochs):
        model.train()
        train_loss=0.0
        train_dice,train_iou=0.0,0.0

        loop=tqdm(train_loader,desc=f'Epoch [{epoch+1}/{cfg.epochs}]')
        for images,masks in loop:
            images,masks= images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            #forward and backward pass with mixed precision
            if use_amp:
                with autocast('cuda'):
                    preds=model(images)
                    loss=criterion(preds,masks)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                preds=model(images)
                loss=criterion(preds,masks)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            #metrics
            dice,iou=calculate_metrics(preds,masks,dice_score,iou_score)
            train_loss+=loss.item()
            train_dice+=dice
            train_iou+=iou

            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        #calculate average metrics for the epoch
        avg_loss=train_loss/len(train_loader)
        avg_dice=train_dice/len(train_loader)
        avg_iou=train_iou/len(train_loader)

        #validation phase
        metrics_fn=lambda preds,masks:calculate_metrics(preds,masks,dice_score,iou_score)
        avg_val_loss,avg_val_dice,avg_val_iou=validate_model(model,test_loader,criterion,device,metrics_fn,scaler)

        #step the scheduler
        scheduler.step()

        #store history
        train_history['loss'].append(avg_loss)
        train_history['dice'].append(avg_dice)
        train_history['iou'].append(avg_iou)
        val_history['loss'].append(avg_val_loss)
        val_history['dice'].append(avg_val_dice)
        val_history['iou'].append(avg_val_iou)

        #print results
        print(f'\nEpoch {epoch+1}/{cfg.epochs}')
        print(f'Train Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}')
        print(f"Val Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}")
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        #save best model
        is_best=model_checkpoint(avg_val_dice, model, epoch)
        if is_best:
            print(f'Best model saved with dice score: {avg_val_dice:.4f}')
            best_dice=avg_val_dice

        #save checkpoint
        save_checkpoint(model,optimizer,scheduler,epoch,train_history,val_history,best_dice,resume_path)

        #early stopping
        if early_stopping(avg_val_loss,model):
            print(f'Early stopping triggered after epoch {epoch+1}')
            print(f'Best validation dice score: {best_dice:.4f}')
            break
    
    print(f'Training completed with best validation dice score: {best_dice:.4f}')


if __name__=="__main__":
    train()
