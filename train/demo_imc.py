import os
import sys

import torch
from tqdm import tqdm

# Add parent directory to path for eval imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from eval.utils.device import to_cpu
from eval.utils.eval_utils import uniform_sample
from sailrecon.models.sail_recon import SailRecon
from sailrecon.utils.load_fn import load_and_preprocess_images

# Import IMC2021 dataset
from datasets.imc2021 import IMC2021, collate_fn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def demo():
    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    _URL = "https://huggingface.co/HKUST-SAIL/SAIL-Recon/resolve/main/sailrecon.pt"
    model = SailRecon(kv_cache=True)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(_URL)
    )
    model = model.to(device=device)
    model.eval()

    # Initialize IMC2021 dataset and dataloader (using hardcoded path from imc2021.py)
    imc_root = "/home/ubuntu/disk6/Motion-from-Structure/release/imc2021/DUSt3R_RoMa"
    max_scenes = 3  # Process only 3 scenes for testing
    
    print(f"Loading IMC2021 dataset from: {imc_root}")
    dataset = IMC2021(root=imc_root, num_images=5)  # Use 5 images for demo
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # One scene per batch
        shuffle=False,  # Keep deterministic for testing
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging
    )
    
    print(f"Dataset contains {len(dataset)} scenes")
    
    # Process each scene from the dataloader
    for scene_idx, batch in enumerate(dataloader):
        if scene_idx >= max_scenes:
            print(f"Processed {max_scenes} scenes, stopping...")
            break
            
        scene_name = batch['scene_name']
        print(f"\n=== Processing Scene {scene_idx + 1}/{min(len(dataset), max_scenes)}: {scene_name} ===")
        
        # Get processed RGB images from the batch (already stacked tensor)
        rgb_processed = batch['rgb_processed']  # Shape: [N, 3, H, W]
        
        if rgb_processed is None or rgb_processed.numel() == 0:
            print(f"No images found in scene {scene_name}, skipping...")
            continue

        print(f"Scene contains {rgb_processed.shape[0]} images")

        # Move to device
        images = rgb_processed.to(device)
        print(f"Stacked images tensor shape: {images.shape}")
        
        # anchor image selection - use all images as anchor images
        anchor_images = images
        print(f"Using all {images.shape[0]} images as anchor images")

        # Create output directory for this scene (hardcoded to /home/ubuntu/tmp)
        out_dir = "/home/ubuntu/tmp"
        scene_output_dir = os.path.join(out_dir, f"scene_{scene_idx:03d}_{scene_name}")
        os.makedirs(scene_output_dir, exist_ok=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # Use original two-phase approach: tmp_forward + reloc
                batch_size = images.shape[0]
                print(f"Images shape: {images.shape}")
                
                # Phase 1: Build scene representation using tmp_forward (all images as anchors)
                print("Phase 1: Building scene representation with tmp_forward...")
                model.tmp_forward(images, fix_rank=300)
                print("Scene representation built successfully")
                
                # Phase 2: Relocalize each image using reloc
                print("Phase 2: Relocalizing each image...")
                predictions = []
                for i in range(batch_size):
                    # Relocalize image i
                    single_image = images[i:i+1]  # Keep batch dimension: (1, 3, H, W)
                    pred = model.reloc(
                        single_image, 
                        fix_rank=300,
                        memory_save=False,  # Keep all predictions
                        save_depth=True,
                        ret_img=True
                    )
                    predictions.append(to_cpu(pred[0]))  # pred is a list with one element
                
                print(f"Relocalization completed, extracted {len(predictions)} predictions")
            
                # save the predicted point cloud and camera poses
                from eval.utils.geometry import save_pointcloud_with_plyfile

                ply_path = os.path.join(scene_output_dir, "pred.ply")
                save_pointcloud_with_plyfile(predictions, ply_path)
                print(f"Saved point cloud to: {ply_path}")

                import numpy as np
                from eval.utils.eval_utils import save_kitti_poses

                poses_w2c_estimated = [
                    one_result["extrinsic"][0].cpu().numpy() for one_result in predictions
                ]
                poses_c2w_estimated = [
                    np.linalg.inv(np.vstack([pose, np.array([0, 0, 0, 1])]))
                    for pose in poses_w2c_estimated
                ]

                poses_path = os.path.join(scene_output_dir, "pred.txt")
                save_kitti_poses(poses_c2w_estimated, poses_path)
                print(f"Saved camera poses to: {poses_path}")
                
                # Save additional information about the scene
                info_path = os.path.join(scene_output_dir, "scene_info.txt")
                with open(info_path, 'w') as f:
                    f.write(f"Scene Name: {scene_name}\n")
                    f.write(f"Number of Images: {images.shape[0]}\n")
                    f.write(f"Number of Correspondences: {len(batch.get('correspondence_indices', []))}\n")
                    f.write(f"Correspondence Pairs: {batch.get('correspondence_indices', [])}\n")
                    f.write(f"Images Tensor Shape: {images.shape}\n")
                    f.write(f"Number of Predictions: {len(predictions)}\n")
                
                print(f"Saved scene info to: {info_path}")
                print(f"Scene {scene_name} processing completed!\n")

    
    print("All scenes processed successfully!")

if __name__ == "__main__":
    demo()
