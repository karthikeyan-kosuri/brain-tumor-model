import torch

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss=val_loss
            self.save_checkpoint(model)
        elif val_loss<self.best_loss-self.min_delta:
            self.best_loss=val_loss
            self.save_checkpoint(model)
            self.counter=0
        else:
            self.counter+=1
        if self.counter>=self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        # store a copy of the state dict to avoid referencing mutable tensors
        self.best_weights = {k: v.clone().cpu() for k, v in model.state_dict().items()}
