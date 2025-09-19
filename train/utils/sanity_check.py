import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from .geometry import compute_relative_pose, backproject_and_reproject
from .io import ImagePreprocessor


def to_device(batch, device):
    """
    Move all tensors in batch to specified device.
    
    Args:
        batch: Batch dictionary containing data
        device: Target device
        
    Returns:
        Dictionary with all tensors moved to device
    """
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def sanity_check_relative_poses(predictions, batch, device, save_dir="/tmp/sanity_check"):
    """
    Sanity check function to compute both ground truth and predicted relative poses,
    recover network estimated depth maps, and verify projection process with visualization.
    
    Args:
        predictions: List of prediction dictionaries from model
        batch: Batch data containing recovery matrices and correspondences
        device: Device for tensor operations
        save_dir: Directory to save visualization plots
    """
    # Move all tensors in batch to device
    batch_on_device = to_device(batch, device)
    
    # Extract predicted poses and intrinsics from model predictions
    predicted_poses = torch.cat([pred['extrinsic'] for pred in predictions], dim=0)  # Shape: [N, 3, 4] or [N, 4, 4]
    predicted_intrinsics = torch.cat([pred['intrinsic'] for pred in predictions], dim=0)  # Shape: [N, 3, 3]
    
    # Get correspondence data (already on device)
    src_idx = batch_on_device['src_idx']        # Shape: [num_pairs]
    dst_idx = batch_on_device['dst_idx']        # Shape: [num_pairs]
    
    # 1. RANDOMLY PICK A PAIR OF IMAGES
    num_pairs = src_idx.shape[0]
    random_pair_idx = torch.randint(0, num_pairs, (1,)).item()
    
    selected_src_idx = src_idx[random_pair_idx].item()
    selected_dst_idx = dst_idx[random_pair_idx].item()
    
    print(f"[Sanity Check] Selected image pair: src={selected_src_idx}, dst={selected_dst_idx}")
    print(f"[Sanity Check] Using predicted intrinsics (recovered to original space)")
    
    # Get poses and intrinsics for selected pair
    src_extrinsic = predicted_poses[selected_src_idx]  # Shape: [3, 4] or [4, 4]
    dst_extrinsic = predicted_poses[selected_dst_idx]  # Shape: [3, 4] or [4, 4]
    relative_pose = compute_relative_pose(src_extrinsic.unsqueeze(0), dst_extrinsic.unsqueeze(0)).squeeze(0)  # Shape: [4, 4]
    
    # 2. RECOVER THE DEPTH BASED ON THE SRC IDX
    depth_pr_processed = batch_on_device['depth_pr_processed']  # Shape: [N, C, H, W]
    image_preprocessor = ImagePreprocessor()
    
    # Get K_prime_to_K matrix for the selected src image
    K_prime_to_K_src = batch_on_device['K_prime_to_K'][selected_src_idx]  # Shape: [3, 3]
    
    # Get processed depth map for the selected src index
    depth_pred_src = depth_pr_processed[selected_src_idx]  # Shape: [C, H, W]
    
    # Recover depth map to original resolution
    depth_recovered = image_preprocessor.reverse_transform_tensor(
        processed_tensor=depth_pred_src,
        K_prime_to_K=K_prime_to_K_src,
        target_size=image_preprocessor.target_size,
        is_depth=True
    )  # Shape: [C, original_height, original_width]
    
    # Remove channel dimension if present
    if depth_recovered.dim() == 3:
        depth_recovered = depth_recovered.squeeze(0)  # Shape: [H, W]
    
    # USE PREDICTED INTRINSICS RECOVERED TO ORIGINAL SPACE
    # Get predicted intrinsics (these are in processed image space, 518x518)
    K_src_predicted = predicted_intrinsics[selected_src_idx]  # Shape: [3, 3] - in processed space
    K_dst_predicted = predicted_intrinsics[selected_dst_idx]  # Shape: [3, 3] - in processed space
    
    # Transform predicted intrinsics from processed space to original space
    # The transformation is: K_original = K_prime_to_K @ K_processed
    K_prime_to_K_dst = batch_on_device['K_prime_to_K'][selected_dst_idx]
    K_src_recovered = K_prime_to_K_src @ K_src_predicted  # Shape: [3, 3] - recovered to original space
    K_dst_recovered = K_prime_to_K_dst @ K_dst_predicted  # Shape: [3, 3] - recovered to original space
    
    # 3. GET CORRESPONDENCE COORDINATES FROM THE RANDOMLY SAMPLED PAIR
    # Use actual predicted correspondences from the network (already in pixel coordinate system)
    src_coords = batch_on_device['src_coords'][random_pair_idx]  # Shape: [K, 2] where K is sample_num
    dst_coords = batch_on_device['dst_coords'][random_pair_idx]  # Shape: [K, 2] - predicted correspondences
    
    # Limit to 10 correspondences for visualization
    num_samples = min(10, src_coords.shape[0])
    indices = torch.randperm(src_coords.shape[0])[:num_samples]
    
    src_coords = src_coords[indices]  # Shape: [10, 2]
    dst_coords = dst_coords[indices]  # Shape: [10, 2]
    
    print(f"[Sanity Check] Using {num_samples} correspondences")
    
    # 4. BILINEAR SAMPLE ON THE RECOVERED DEPTH MAP AT CORRESPONDENCE LOCATIONS
    H, W = depth_recovered.shape
    
    # Normalize correspondence coordinates for grid_sample (range [-1, 1])
    # src_coords is in pixel coordinates [u, v], need to normalize to [-1, 1]
    grid_u = (src_coords[:, 0] / (W - 1)) * 2.0 - 1.0  # Normalize to [-1, 1]
    grid_v = (src_coords[:, 1] / (H - 1)) * 2.0 - 1.0  # Normalize to [-1, 1]
    grid = torch.stack([grid_u, grid_v], dim=1).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, num_samples, 2]
    
    # Bilinear sample depths at correspondence locations
    depth_map_for_sampling = depth_recovered.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    sampled_depths = F.grid_sample(
        depth_map_for_sampling, 
        grid, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=False
    ).squeeze().squeeze()  # Shape: [num_samples]
    

    
    # 5. APPLY THE PROJECTION FUNCTION TO GET PROJECTED DST PIXEL LOCATIONS
    # Prepare tensors for backproject_and_reproject function using RECOVERED intrinsics
    src_coords_batch = src_coords.unsqueeze(0)  # Shape: [1, 10, 2]
    src_depth_batch = sampled_depths.unsqueeze(0)  # Shape: [1, 10]
    src_intrinsic_batch = K_src_recovered.unsqueeze(0)  # Shape: [1, 3, 3]
    dst_intrinsic_batch = K_dst_recovered.unsqueeze(0)  # Shape: [1, 3, 3]
    relative_pose_batch = relative_pose.unsqueeze(0)  # Shape: [1, 4, 4]
    src_depth_scale = torch.ones(1, 1, device=device, dtype=torch.float32)  # Shape: [1, 1]
    
    # Project to destination image
    projected_dst_coords, valid_mask = backproject_and_reproject(
        src_coords=src_coords_batch,
        src_depth=src_depth_batch,
        src_intrinsic=src_intrinsic_batch,
        dst_intrinsic=dst_intrinsic_batch,
        relative_pose=relative_pose_batch,
        src_depth_scale=src_depth_scale
    )  # Shape: [1, 10, 2], [1, 10]
    
    projected_dst_coords = projected_dst_coords.squeeze(0)  # Shape: [10, 2]
    valid_mask = valid_mask.squeeze(0)  # Shape: [10]
    
    # Print key results only
    mean_offset = (dst_coords - projected_dst_coords).mean(dim=0).detach().cpu().numpy()
    print(f"[Sanity Check] Mean projection offset: [{mean_offset[0]:.1f}, {mean_offset[1]:.1f}] pixels")
    
    # 6. USE MATPLOTLIB TO PLOT BOTH RGB IMAGES
    # Get original RGB images
    rgb_processed = batch['rgb_processed']  # Shape: [N, C, H, W]
    
    # Recover RGB images to original resolution
    rgb_src_processed = rgb_processed[selected_src_idx]  # Shape: [C, H_proc, W_proc]
    rgb_dst_processed = rgb_processed[selected_dst_idx]  # Shape: [C, H_proc, W_proc]
    
    # Recover RGB images to original resolution
    rgb_src_recovered = image_preprocessor.reverse_transform_tensor(
        processed_tensor=rgb_src_processed,
        K_prime_to_K=K_prime_to_K_src,
        target_size=image_preprocessor.target_size,
        is_depth=False
    )  # Shape: [C, original_height, original_width]
    
    K_prime_to_K_dst = batch_on_device['K_prime_to_K'][selected_dst_idx]
    rgb_dst_recovered = image_preprocessor.reverse_transform_tensor(
        processed_tensor=rgb_dst_processed,
        K_prime_to_K=K_prime_to_K_dst,
        target_size=image_preprocessor.target_size,
        is_depth=False
    )  # Shape: [C, original_height, original_width]
    
    # Convert tensors to numpy for matplotlib (CHW -> HWC)
    rgb_src_np = rgb_src_recovered.permute(1, 2, 0).cpu().numpy()
    rgb_dst_np = rgb_dst_recovered.permute(1, 2, 0).cpu().numpy()
    
    # Convert coordinates to numpy (detach gradients first)
    src_coords_np = src_coords.detach().cpu().numpy()
    dst_coords_np = dst_coords.detach().cpu().numpy()  # Add destination correspondences
    projected_dst_coords_np = projected_dst_coords.detach().cpu().numpy()
    valid_mask_np = valid_mask.detach().cpu().numpy()
    
    # 7. SAVE VISUALIZATION
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 7))
    
    # Plot source image with sample points
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_src_np)
    plt.scatter(src_coords_np[:, 0], src_coords_np[:, 1], c='red', s=50, marker='o', alpha=0.8)
    for i, (u, v) in enumerate(src_coords_np):
        plt.annotate(f'{i}', (u, v), xytext=(5, 5), textcoords='offset points', 
                    color='white', fontweight='bold', fontsize=8)
    plt.title(f'Source Image (idx={selected_src_idx})\nRed points: sampled locations')
    plt.axis('off')
    
    # Plot destination image with correspondences and projected points
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_dst_np)
    
    # Plot destination correspondences in blue
    plt.scatter(dst_coords_np[:, 0], dst_coords_np[:, 1], c='blue', s=50, marker='s', alpha=0.8, label='Correspondences')
    for i, (u, v) in enumerate(dst_coords_np):
        plt.annotate(f'{i}', (u, v), xytext=(5, 5), textcoords='offset points', 
                    color='white', fontweight='bold', fontsize=8)
    
    # Plot valid projections in green (only valid ones)
    valid_coords = projected_dst_coords_np[valid_mask_np.astype(bool)]
    
    if len(valid_coords) > 0:
        plt.scatter(valid_coords[:, 0], valid_coords[:, 1], c='green', s=50, marker='o', alpha=0.8, label='Projections')
        for i, coord in enumerate(valid_coords):
            valid_idx = np.where(valid_mask_np)[0][i]
            plt.annotate(f'{valid_idx}', (coord[0], coord[1]), xytext=(5, -15), 
                        textcoords='offset points', color='yellow', fontweight='bold', fontsize=8)
    
    plt.title(f'Destination Image (idx={selected_dst_idx})\nBlue squares: correspondences, Green circles: projections (pred K)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    scene_name = batch.get('scene_name', 'unknown_scene')
    save_path = os.path.join(save_dir, f"projection_check_{scene_name}_{selected_src_idx}_to_{selected_dst_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Sanity Check] Projection visualization saved to: {save_path}")
    print(f"  Valid projections: {valid_mask_np.sum()}/{num_samples}")
    
    # 8. VISUALIZE RECOVERED DEPTH MAP SIDE BY SIDE WITH RGB
    plt.figure(figsize=(15, 7))
    
    # Left: Source RGB image  
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_src_np)
    plt.title(f'Source RGB Image (idx={selected_src_idx})')
    plt.axis('off')
    
    # Right: Recovered depth map
    plt.subplot(1, 2, 2) 
    depth_np = depth_recovered.cpu().numpy()
    
    # Mask invalid depths (zeros) for better visualization
    depth_masked = np.where(depth_np > 0, depth_np, np.nan)
    
    # Use viridis colormap for depth visualization
    valid_depths = depth_np[depth_np > 0]
    if len(valid_depths) > 0:
        vmin, vmax = np.percentile(valid_depths, [5, 95])  # 5th-95th percentile for better contrast
        im = plt.imshow(depth_masked, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, shrink=0.8, label='Depth (meters)')
    else:
        plt.imshow(depth_masked, cmap='viridis')
        
    # Overlay correspondence points
    plt.scatter(src_coords_np[:, 0], src_coords_np[:, 1], c='red', s=30, alpha=0.8, edgecolors='white', linewidth=1)
    plt.title(f'Recovered Depth Map (idx={selected_src_idx})')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save depth visualization
    depth_save_path = os.path.join(save_dir, f"depth_visualization_{scene_name}_{selected_src_idx}.png")
    plt.savefig(depth_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Sanity Check] Depth visualization saved to: {depth_save_path}")

