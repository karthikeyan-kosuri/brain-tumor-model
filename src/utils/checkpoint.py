import numpy as np
import torch
import os
import re
from pathlib import Path
from configs import seg_config as cfg
from src.model.unet import build_unet

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_dice',mode='max'):
        self.filepath=filepath
        self.monitor=monitor
        self.mode=mode
        self.best_score=None

        if mode=='max':
            self.monitor_op=np.greater
            self.best_score=-np.inf
        else:
            self.monitor_op=np.less
            self.best_score=np.inf
    
    def __call__(self,current_score,model, epoch):
        if self.monitor_op(current_score,self.best_score):
            self.best_score=current_score
            filepath = self.filepath.format(epoch=epoch+1, **{self.monitor: current_score})
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_score': self.best_score}, filepath)
            return True
        return False

def resume_training(model, optimizer,scheduler,checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f'Resuming training from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_dice=checkpoint.get('best_dice',0.0)
        train_history=checkpoint.get('train_history',{'loss':[],'dice':[],'iou':[]})
        val_history=checkpoint.get('val_history',{'loss':[],'dice':[],'iou':[]})

        return start_epoch, best_dice, train_history, val_history

    train_history={'loss':[],'dice':[],'iou':[]}
    val_history={'loss':[],'dice':[],'iou':[]}
    return 0, 0.0, train_history, val_history

def save_checkpoint(model, optimizer, scheduler, epoch, train_history, val_history, best_dice, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_history': train_history,
        'val_history': val_history,
        'best_dice': best_dice
    }
    torch.save(checkpoint, filepath)


def extract_dice_from_filename(filename):
    match = re.search(r"dice_([0-9.]+)", filename)
    if match:
        return float(match.group(1))
    return None

def find_best_checkpoint(checkpoint_dir='outputs/checkpoints', monitor='val_dice'):
    checkpoint_dir=Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint_dir}')
    
    best_checkpoint=None
    best_score=-float('inf')

    for ckpt_file in checkpoint_dir.glob('*.pth'):
        score=extract_dice_from_filename(ckpt_file.name)
        if score is not None and score>best_score:
            best_score=score
            best_checkpoint=ckpt_file
        
    if best_checkpoint:
        print(f'Best checkpoint found at: {best_checkpoint} with {monitor}={best_score:.4f}')
        return best_checkpoint
    else:
        raise FileNotFoundError('No valid checkpoint files found in the directory.')

def load_model(checkpoint_path=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If checkpoint path is None or a directory → auto-select best checkpoint
    if checkpoint_path is None or os.path.isdir(checkpoint_path):
        checkpoint_path = find_best_checkpoint(checkpoint_path or "outputs/checkpoints")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_unet(in_c=cfg.in_channel, out_c=cfg.out_channel).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

