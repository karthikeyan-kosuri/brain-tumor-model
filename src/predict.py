import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.utils.checkpoint import load_model
from src.utils.io_utils import setup_prediction_directory,save_mask,save_overlay,save_heatmap
from src.dataset.preprocess_utils import preprocess_image

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def compute_confidence(prob_map, binary_mask):
    avg_conf=float(prob_map.mean())
    max_conf=float(prob_map.max())
    tumor_pixels=int(binary_mask.sum())
    total_pixels=binary_mask.size
    tumor_percent=(tumor_pixels/total_pixels)*100
    tumor_conf=float(prob_map[binary_mask>0].mean()) if tumor_pixels>0 else 0.0

    return {'avg_confidence':avg_conf, 'max_confidence': max_conf, 'tumor_pixels':tumor_pixels, 'total_pixels':total_pixels, 'tumor_percent':tumor_percent, 'tumor_confidence':tumor_conf}

def predict_single(model,image_path,output_dir,threshold=0.5,model_input_size=(256,256)):
    
    img_tensor,orig_image,orig_size=preprocess_image(image_path,model_input_size)
    img_tensor=img_tensor.to(device)

    with torch.no_grad():
        logits=model(img_tensor)
        probs=torch.sigmoid(logits).squeeze().cpu().numpy()
        binary_mask=(probs>threshold).astype(np.uint8)

    #resize back to original size
    probs_resized = np.array(Image.fromarray((probs*255).astype(np.uint8)).resize(orig_size, Image.BILINEAR))/255.0
    binary_resized = np.array(Image.fromarray((binary_mask*255).astype(np.uint8)).resize(orig_size, Image.NEAREST))
    binary_resized = (binary_resized / 255).astype(np.uint8)

    metrics=compute_confidence(probs_resized,binary_resized)

    #save outputs
    image_name=Path(image_path).stem
    save_mask(binary_resized,output_dir['masks']/f"{image_name}_mask.png")
    save_heatmap(probs_resized,output_dir['heatmaps']/f"{image_name}_heatmap.png")
    save_overlay(orig_image,binary_resized,probs_resized,output_dir['overlays']/f"{image_name}_overlay.png",metrics,threshold)
    return metrics

def predict_folder(model,folder_path,output_dir,threshold=0.5,model_input_size=(256,256)):
    folder_path=Path(folder_path)
    image_files=list(folder_path.glob("*.png"))+list(folder_path.glob("*.jpg"))+list(folder_path.glob("*.jpeg"))
    if not image_files:
        raise ValueError("No image files found in the specified folder.")
    
    all_metrics=[]
    for img_path in tqdm(image_files,desc='Predicting images'):
        metrics=predict_single(model,str(img_path),output_dir,threshold,model_input_size)
        print(f"{img_path.name}: tumor area {metrics['tumor_percent']:.2f}%, avg confidence {metrics['avg_confindence']:.3f} ")
        all_metrics.append(metrics)

    return all_metrics

def main():
    parser=argparse.ArgumentParser(description="Brain tumor prediction using MRI images")
    input_group=parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image',type=str,help="Path to a single image file")
    input_group.add_argument('--folder',type=str,help="Path to a folder with images")
    parser.add_argument('--checkpoint',type=str,default=None,help="Path to model checkpoint")
    parser.add_argument('--model-size',type=int,nargs=2,default=[256,256],help="Model input image size (height,width)")
    parser.add_argument('--threshold',type=float,default=0.5,help="Threshold for binary mask generation")
    args=parser.parse_args()

    model=load_model(args.checkpoint,device)
    output_dirs=setup_prediction_directory('outputs/results/predictions')
    
    if args.image:
        predict_single(model,args.image,output_dirs,args.threshold,tuple(args.model_size))
        print(f"Predictions saved in {output_dirs['overlays'].parent}")
    elif args.folder:
        predict_single(model,args.folder,output_dirs,args.threshold,tuple(args.model_size))
        print(f"Predictions saved in {output_dirs['overlays'].parent}")

if __name__=='__main__':
    main()
