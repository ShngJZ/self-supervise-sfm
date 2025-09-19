import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple
from .geometry import to_homogeneous, from_homogeneous, pad_poses, unpad_poses

def monodepth2vls(monodepth: torch.Tensor, vmax: float = 0.18, 
                  percentile: Optional[int] = None, viewind: int = 0) -> Image.Image:
    """Convert monodepth to visualization using magma colormap.
    
    Args:
        monodepth: Depth tensor with shape (B, C, H, W) or (B, H, W)
        vmax: Maximum depth value for normalization
        percentile: If provided, use percentile for vmax calculation
        viewind: Index of view to visualize
        
    Returns:
        PIL Image with depth visualization
    """
    colormap = plt.get_cmap('magma')
    
    # Convert to numpy and select view
    if isinstance(monodepth, torch.Tensor):
        if monodepth.ndim == 3:
            monodepth = monodepth.unsqueeze(1)
        depth_np = monodepth[viewind, 0].detach().cpu().numpy()
    else:
        depth_np = monodepth
    
    # Calculate vmax from percentile if specified
    if percentile is not None:
        valid_depths = depth_np[depth_np > 0]
        vmax = np.percentile(valid_depths, 95) if len(valid_depths) > 100 else 1.0
    
    # Normalize and apply colormap
    depth_normalized = np.clip(depth_np / vmax, 0, 1)
    depth_colored = (colormap(depth_normalized) * 255).astype(np.uint8)
    
    return Image.fromarray(depth_colored[:, :, :3])

def image2vls(image: torch.Tensor, viewind: int = 0) -> Image.Image:
    """Convert image tensor to PIL Image for visualization.
    
    Args:
        image: Image tensor with shape (B, C, H, W)
        viewind: Index of view to visualize
        
    Returns:
        PIL Image
    """
    if isinstance(image, torch.Tensor):
        # Convert from (B, C, H, W) to (B, H, W, C) and select view
        image_np = image.detach().cpu().permute(0, 2, 3, 1)[viewind].numpy()
    else:
        image_np = image
    
    # Normalize to [0, 255] range
    if image_np.max() <= 1.0001:
        image_np = image_np * 255.0
    
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)

def corres2vls(rgb_src: torch.Tensor, rgb_dst: torch.Tensor, corres: torch.Tensor, 
               certainty: torch.Tensor, viewind: int = 0) -> Image.Image:
    """Visualize correspondences between source and destination images.
    
    Args:
        rgb_src: Source RGB image with shape (B, C, H, W)
        rgb_dst: Destination RGB image with shape (B, C, H, W)
        corres: Correspondence coordinates with shape (B, H, W, 4) [src_x, src_y, dst_x, dst_y]
        certainty: Correspondence certainty with shape (B, 1, H, W)
        viewind: Index of view to visualize
        
    Returns:
        PIL Image showing [source_sampled | warped_dest | reference_dest]
    """
    _, _, height, width = rgb_src.shape
    
    # Split correspondences into source and destination coordinates
    corres_src, corres_dst = torch.split(corres, [2, 2], dim=-1)
    
    # Sample RGB values at correspondence locations
    rgb_src_sampled = F.grid_sample(rgb_src, corres_src, mode="bilinear", align_corners=False)
    rgb_dst_ref = F.grid_sample(rgb_dst, corres_src, mode="bilinear", align_corners=False)
    rgb_dst_sampled = F.grid_sample(rgb_dst, corres_dst, mode="bilinear", align_corners=False)
    
    # Mark invalid correspondences (low certainty) as white
    invalid_mask = (certainty < 1e-3).expand(-1, 3, -1, -1)
    rgb_dst_sampled[invalid_mask] = 1.0
    
    # Convert to PIL images and resize to original dimensions
    src_img = image2vls(rgb_src_sampled, viewind=viewind).resize((width, height))
    dst_img = image2vls(rgb_dst_sampled, viewind=viewind).resize((width, height))
    ref_img = image2vls(rgb_dst_ref, viewind=viewind).resize((width, height))
    
    # Concatenate horizontally: [source | warped_dest | reference]
    combined = np.concatenate([np.array(src_img), np.array(dst_img), np.array(ref_img)], axis=1)
    
    return Image.fromarray(combined)

def tuple2vls(rgb_src: torch.Tensor, rgb_dst: torch.Tensor, 
              intrinsic_src: torch.Tensor, intrinsic_dst: torch.Tensor, 
              pose: torch.Tensor, sampled_src: torch.Tensor, sampled_dst: torch.Tensor,
              depth_src_f: torch.Tensor, depth_dst_f: torch.Tensor, 
              nlim: Optional[int] = 20, sv_path: Optional[str] = None) -> None:
    """Visualize geometric correspondences with 3D reprojection validation.
    
    Args:
        rgb_src: Source RGB image
        rgb_dst: Destination RGB image  
        intrinsic_src: Source camera intrinsic matrix
        intrinsic_dst: Destination camera intrinsic matrix
        pose: Relative pose transformation matrix
        sampled_src: Source 2D sample points
        sampled_dst: Destination 2D sample points
        depth_src_f: Depth values at source points
        depth_dst_f: Depth values at destination points
        nlim: Number of points to visualize (None for all points)
        sv_path: Optional path to save the visualization
    """
    # Limit number of points for visualization
    if nlim is not None:
        # Use specific range for debugging (can be changed to [0:nlim])
        start_idx, end_idx = 4623, 4625
        sampled_src = sampled_src[start_idx:end_idx]
        sampled_dst = sampled_dst[start_idx:end_idx]
        depth_src_f = depth_src_f[start_idx:end_idx]
        depth_dst_f = depth_dst_f[start_idx:end_idx]
        nlim = end_idx - start_idx
    else:
        nlim = len(depth_src_f)

    # Generate random colors for each point
    point_colors = np.random.rand(nlim, 3)

    # Project source points to destination via 3D transformation
    pts3d_cam_src = to_homogeneous(sampled_src) * depth_src_f.view(nlim, 1) @ torch.inverse(intrinsic_src).T
    pts3d_world = to_homogeneous(pts3d_cam_src) @ pose.T
    pts2d_dst_proj = from_homogeneous(pts3d_world @ intrinsic_dst.T)

    # Project destination points to source via inverse 3D transformation  
    pts3d_cam_dst = to_homogeneous(sampled_dst) * depth_dst_f.view(nlim, 1) @ torch.inverse(intrinsic_dst).T
    pose_inv = unpad_poses(pad_poses(pose).inverse())
    pts3d_world_inv = to_homogeneous(pts3d_cam_dst) @ pose_inv.T
    pts2d_src_proj = from_homogeneous(pts3d_world_inv @ intrinsic_src.T)

    # Create visualization with 3 rows and 2 columns
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
    fig.suptitle('Geometric Correspondence Validation', fontsize=16)
    
    def plot_points_on_image(ax, image, points, colors, title):
        """Helper function to plot points on image."""
        ax.imshow(image)
        ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), 
                  s=50, c=colors, alpha=0.8, edgecolors='white', linewidth=1)
        ax.set_title(title)
        ax.axis('off')
    
    # Row 1: Original correspondences
    plot_points_on_image(axes[0, 0], rgb_src, sampled_src, point_colors, 'Source Image (Original Points)')
    plot_points_on_image(axes[0, 1], rgb_dst, sampled_dst, point_colors, 'Destination Image (Original Points)')
    
    # Row 2: Source to destination projection
    plot_points_on_image(axes[1, 0], rgb_src, sampled_src, point_colors, 'Source Points')
    plot_points_on_image(axes[1, 1], rgb_dst, pts2d_dst_proj, point_colors, 'Projected to Destination')
    
    # Row 3: Destination to source projection  
    plot_points_on_image(axes[2, 0], rgb_dst, sampled_dst, point_colors, 'Destination Points')
    plot_points_on_image(axes[2, 1], rgb_src, pts2d_src_proj, point_colors, 'Projected to Source')
    
    plt.tight_layout()
    
    # Save if path provided
    if sv_path is not None:
        plt.savefig(sv_path, bbox_inches='tight', pad_inches=0, dpi=300)
        
    plt.close()

def plot_cdf_pdf_curves(frame_cdfs: torch.Tensor, frame_pdfs: torch.Tensor, 
                        min_val: float, max_val: float, num_bins: int,
                        num_frames_to_plot: int = 4, save_path: Optional[str] = None) -> None:
    """Plot CDF and PDF curves for randomly selected frames.
    
    Args:
        frame_cdfs: Frame CDF values with shape [num_frames, num_bins]
        frame_pdfs: Frame PDF values with shape [num_frames, num_bins] 
        min_val: Minimum value of the histogram range
        max_val: Maximum value of the histogram range
        num_bins: Number of histogram bins
        num_frames_to_plot: Number of random frames to plot (default: 4)
        save_path: Optional path to save the plot
    """
    # Convert to numpy
    if isinstance(frame_cdfs, torch.Tensor):
        frame_cdfs = frame_cdfs.detach().cpu().numpy()
    if isinstance(frame_pdfs, torch.Tensor):
        frame_pdfs = frame_pdfs.detach().cpu().numpy()
    
    num_frames = frame_cdfs.shape[0]
    if num_frames == 0:
        print("Warning: No frames to plot")
        return
        
    # Randomly select frames to plot
    frame_indices = np.random.choice(num_frames, size=min(num_frames_to_plot, num_frames), replace=False)
    
    # Create bin centers for x-axis
    bin_centers = np.linspace(min_val, max_val, num_bins)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generate distinct colors for each frame
    colors = plt.cm.Set1(np.linspace(0, 1, len(frame_indices)))
    
    # Plot CDFs
    for i, frame_idx in enumerate(frame_indices):
        ax1.plot(bin_centers, frame_cdfs[frame_idx], 
                color=colors[i], linewidth=2, alpha=0.8, 
                label=f'Frame {frame_idx}')
    
    ax1.set_xlabel('Residual Value (log(1+x))')
    ax1.set_ylabel('CDF')
    ax1.set_title('Cumulative Distribution Function (CDF)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Plot PDFs
    for i, frame_idx in enumerate(frame_indices):
        ax2.plot(bin_centers, frame_pdfs[frame_idx], 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'Frame {frame_idx}')
    
    ax2.set_xlabel('Residual Value (log(1+x))')
    ax2.set_ylabel('PDF')
    ax2.set_title('Probability Density Function (PDF)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Saved CDF/PDF plot to: {save_path}")
        
    plt.close()

