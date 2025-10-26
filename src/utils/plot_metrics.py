import torch
import matplotlib.pyplot as plt
import os

def plot_checkpoint_metrics(checkpoint_path='outputs/checkpoints/last_checkpoint.pth', save_dir='outputs/plots'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract histories with defaults
    train_history = ckpt.get('train_history', {})
    val_history = ckpt.get('val_history', {})
    best_dice = ckpt.get('best_dice', None)
    
    # Validate that we have data
    required_keys = ['dice', 'iou', 'loss']
    for key in required_keys:
        if key not in train_history or key not in val_history:
            raise ValueError(f"Missing '{key}' in checkpoint history")
    
    if len(train_history['dice']) == 0:
        raise ValueError("No training history found in checkpoint")
    
    # Print a summary of metrics
    print("Loaded checkpoint:", checkpoint_path)
    print(f"Epochs trained: {len(train_history['dice'])}")
    if best_dice is not None:
        print(f"Best validation Dice score: {best_dice:.4f}")
    
    print("\nMetrics calculated:")
    for i in range(len(train_history['dice'])):
        print(f"  Epoch {i+1}: "
              f"Train Dice={train_history['dice'][i]:.4f}, "
              f"Val Dice={val_history['dice'][i]:.4f}, "
              f"Train IoU={train_history['iou'][i]:.4f}, "
              f"Val IoU={val_history['iou'][i]:.4f}")
    
    epochs = range(1, len(train_history['dice']) + 1)
    
    # --- Dice ---
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_history['dice'], label='Train Dice', linewidth=2)
    plt.plot(epochs, val_history['dice'], label='Val Dice', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Score Over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dice_curve.png'), dpi=300)
    plt.close()
    
    # --- IoU ---
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_history['iou'], label='Train IoU', linewidth=2)
    plt.plot(epochs, val_history['iou'], label='Val IoU', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('IoU Over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iou_curve.png'), dpi=300)
    plt.close()
    
    # --- Loss ---
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_history['loss'], label='Train Loss', linewidth=2)
    plt.plot(epochs, val_history['loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    print(f"\nPlots saved in: {save_dir}")

if __name__ == "__main__":
    plot_checkpoint_metrics()