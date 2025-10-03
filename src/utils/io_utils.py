import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

def setup_directories():
    directories=['outputs/checkpoints','outputs/results']
    for directory in directories:
        os.makedirs(directory,exist_ok=True)

def setup_prediction_directory(base_dir):
    dirs={'masks':Path(base_dir)/'masks', 'overlays': Path(base_dir)/'overlays', 'heatmaps': Path(base_dir)/'heatmaps'}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs

def save_mask(mask_np,save_path):
    img=Image.fromarray((mask_np*255).astype(np.uint8))
    img.save(save_path)

def save_heatmap(prob_map, save_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    plt.colorbar(label='Tumor Probability')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_overlay(original_image,binary_mask,prob_map,save_path,metrics,threshold=0.5):
    fig,axes=plt.subplots(1,4, figsize=(20,5))

    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title(f'Binary Mask\n(Threshold: {threshold})')
    axes[1].axis('off')

    im = axes[2].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('Probability Heatmap')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    axes[3].imshow(original_image, cmap='gray')
    axes[3].imshow(binary_mask, cmap='Reds', alpha=0.4)
    axes[3].set_title('Overlay')
    axes[3].axis('off')

    metrics_text = (
    f"Avg confidence: {metrics['avg_confidence']:.3f}\n"
    f"Tumor confidence: {metrics['tumor_confidence']:.3f}\n"
    f"Tumor area: {metrics['tumor_percent']:.3f}\n"
    f"Tumor pixels: {metrics['tumor_pixels']:.3f}"
    )
    fig.text(0.5,0.22,metrics_text,ha='center', fontsize=10, bbox=dict(boxstyle="round",facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path,dpi=150,bbox_inches='tight')
    plt.close()