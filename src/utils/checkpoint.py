import numpy as np
import torch
import os

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
