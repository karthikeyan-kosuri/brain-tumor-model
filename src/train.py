import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from configs import seg_config as cfg
from src.dataset.brisc_dataset import BrainMRIDataset
from src.utils.metrics import dice_score, iou_score
from src.model.unet import build_unet
from tqdm import tqdm

def train():
    #loading the dataset
    train_dataset=BrainMRIDataset(cfg.train_images, cfg.train_mask)
    test_dataset= BrainMRIDataset(cfg.test_images, cfg.test_mask)
    train_loader=DataLoader( train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_worker, pin_memory=True)
    test_loader=DataLoader( test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_worker, pin_memory=True)

    #loading the model with loss optimizer functions
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_unet(in_c= cfg.in_channel, out_c= cfg.out_channel).to(device)
    criterion=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters(), lr=cfg.learning_rate)

    #training loop
    for epoch in range(cfg.epochs):
        model.train()
        train_loss=0.0
        train_dice,train_iou=0.0,0.0

        loop=tqdm(train_loader,desc=f'Epoch [{epoch+1}/{cfg.epochs}]')
        for images,masks in loop:
            images,masks= images.to(device), masks.to(device)

            #forward pass
            preds=model(images)
            loss=criterion(preds,masks)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #metrics
            preds_sigmoid=torch.sigmoid(preds)
            train_loss+=loss.item()
            train_dice+=dice_score(preds_sigmoid,masks)
            train_iou+=iou_score(preds_sigmoid,masks)

            loop.set_postfix(loss=loss.item())
        
        avg_loss=train_loss/len(train_loader)
        avg_dice=train_dice/len(train_loader)
        avg_iou=train_iou/len(train_loader)

        print(f'\nEpoch {epoch+1}/{cfg.epochs}')
        print(f'Train Loss: {avg_loss:.4f}')
        print(f'Train Dice loss: {avg_dice: .4f}')
        print(f'Train IoU score: {avg_iou: .4f}')

        model.eval()
        val_loss=0.0
        val_dice,val_iou=0.0,0.0
        with torch.no_grad():
            for images,masks in test_loader:
                images,masks= images.to(device), masks.to(device)
                preds=model(images)
                loss=criterion(preds,masks)

                preds_sigmoid=torch.sigmoid(preds)
                val_loss+=loss.item()
                val_dice+=dice_score(preds_sigmoid,masks)
                val_iou+=iou_score(preds_sigmoid,masks)

        avg_val_loss=val_loss/len(test_loader)
        avg_val_dice=val_dice/len(test_loader)
        avg_val_iou=val_iou/len(test_loader)

        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Dice score: {avg_val_dice:.4f}')
        print(f'Validation IoU score: {avg_val_iou:.4f}')

        torch.save(model.state_dict(),f'outputs/checkpoints/unet_epoch{epoch+1}.pth')

if __name__=="__main__":
    train()
