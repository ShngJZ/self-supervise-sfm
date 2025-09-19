import os
import glob
import random
import h5py
import torch
import numpy as np
import PIL.Image
import io
from collections import OrderedDict
from typing import List, Dict, Any
from natsort import natsorted

import sys
train_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, train_dir)
from utils.io import ImagePreprocessor, sample_correspondence_and_depth


class IMC2021(torch.utils.data.Dataset):
    def __init__(self, root: str, sample_num: int = 10000, min_corres_conf: float = 0.1, num_images: int = 5):
        """
        IMC2021 Dataset for multi-GPU training with HDF5 files.
        
        Args:
            root: Root directory containing scene folders
            sample_num: Number of correspondence points to sample per pair
            min_corres_conf: Minimum confidence threshold for filtering correspondences
            num_images: Number of images to randomly select per scene
        """
        super().__init__()
        self.root = root
        self.sample_num = sample_num
        self.min_corres_conf = min_corres_conf
        self.num_images = num_images
        self.shared_focal = False
        
        # Initialize ImagePreprocessor with default target_size=518
        self.image_preprocessor = ImagePreprocessor()
        
        # Find all scene folders
        self.scene_folders = self._find_scene_folders()
        print(f"[IMC2021] Found {len(self.scene_folders)} scenes")

        random.shuffle(self.scene_folders)
        
    def _find_scene_folders(self) -> List[str]:
        """Find all scene folders containing HDF5 files."""
        scene_folders = []
        
        # Get all directories in root
        for item in os.listdir(self.root):
            folder_path = os.path.join(self.root, item)
            if os.path.isdir(folder_path):
                # Check if folder contains HDF5 file
                hdf5_files = glob.glob(os.path.join(folder_path, "*.hdf5"))
                if hdf5_files:
                    scene_folders.append(folder_path)
        
        return sorted(scene_folders)
    
    def _get_hdf5_path(self, scene_folder: str) -> str:
        """Get the HDF5 file path for a scene folder."""
        hdf5_files = glob.glob(os.path.join(scene_folder, "*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 file found in {scene_folder}")
        return hdf5_files[0]  # Take the first HDF5 file
    
    def _png2coords(self, coords: np.ndarray) -> np.ndarray:
        """Convert PNG coordinates to normalized coordinates."""
        uint16max = 65535.0
        return coords.astype(np.float32) / uint16max * 2 - 1
    
    def _png2certainty(self, certainty: np.ndarray) -> np.ndarray:
        """Convert PNG certainty to normalized certainty."""
        return certainty.astype(np.float32) / 1000
    
    def _read_hdf5_image(self, hdf5_file, group_key: str, image_name: str) -> PIL.Image.Image:
        """Unified method to read any image from HDF5 file as PIL Image."""
        return PIL.Image.open(io.BytesIO(np.array(hdf5_file[group_key][image_name])))
    
    def _read_camera_intrinsics(self, hdf5_file, image_name: str) -> torch.Tensor:
        """Read camera intrinsics from HDF5 file."""
        # Convert .jpg to .txt for intrinsics lookup
        image_txt = image_name.replace('.jpg', '.txt')
        K = np.array(hdf5_file['intrinsic_gt'][image_txt])
        return torch.from_numpy(K.astype(np.float32))
    
    def _read_camera_pose(self, hdf5_file, image_name: str) -> torch.Tensor:
        """
        Read camera pose (world-to-camera) from HDF5 file.
        
        Args:
            hdf5_file: Open HDF5 file handle
            image_name: Image name (e.g., '000000.jpg')
            
        Returns:
            Camera pose as torch.Tensor (world-to-camera transformation)
        """
        # Convert .jpg to .txt for pose lookup (same convention as intrinsics)
        idx_name = image_name.replace('.jpg', '')  # Remove extension
        pose_txt = f"{idx_name}.txt"
        
        # Read pose from 'pose_w2c_gt' group
        pose_data = np.array(hdf5_file['pose_w2c_gt'][pose_txt])
        return torch.from_numpy(pose_data.astype(np.float32))
    
    def _read_corres(self, hdf5_file, pair_name: str):
        """Read correspondence data from HDF5 file."""
        h5group = hdf5_file['corres_i2j'][pair_name]
        
        # Read correspondence images using direct PIL access (h5group acts as file-like)
        coordsjx = np.array(PIL.Image.open(io.BytesIO(np.array(h5group[f"{pair_name}_x.png"]))))
        coordsjy = np.array(PIL.Image.open(io.BytesIO(np.array(h5group[f"{pair_name}_y.png"]))))
        certainty = np.array(PIL.Image.open(io.BytesIO(np.array(h5group[f"{pair_name}_conf.png"]))))
        
        hs, ws = certainty.shape
        
        # Convert PNG to correspondence coordinates
        coordsjx = self._png2coords(coordsjx)
        coordsjy = self._png2coords(coordsjy)
        certainty = self._png2certainty(certainty)
        
        coords_dst = np.stack([coordsjx, coordsjy], axis=-1)
        
        # Generate source coordinates
        xx, yy = np.meshgrid(
            np.linspace(-1 + 1 / ws, 1 - 1 / ws, ws), 
            np.linspace(-1 + 1 / hs, 1 - 1 / hs, hs), 
            indexing="xy"
        )
        coords_src = np.stack([xx, yy], axis=-1)
        
        return coords_src, coords_dst, certainty
    
    def _get_available_images(self, hdf5_path: str) -> List[str]:
        """Get list of available image names from HDF5 file."""
        with h5py.File(hdf5_path, 'r') as f:
            return list(f['rgb'].keys())
    
    def _read_selected_hdf5_data(self, hdf5_path: str, selected_images: List[str]) -> Dict[str, Any]:
        """Read only selected data from HDF5 file for efficiency."""
        data = {
            'rgb': {},
            'depth_pr': {},
            'corres_i2j': {},
            'intrinsics': {},
            'poses_w2c': {}
        }
        
        with h5py.File(hdf5_path, 'r') as f:
            # Read selected images and their associated data
            for img_name in selected_images:
                # Read RGB image using unified method
                data['rgb'][img_name] = self._read_hdf5_image(f, 'rgb', img_name)
                
                # Read depth prediction using unified method
                image_name_png = img_name.replace('.jpg', '.png')
                data['depth_pr'][img_name] = self._read_hdf5_image(f, 'depth_pr', image_name_png)
                
                # Read camera intrinsics
                data['intrinsics'][img_name] = self._read_camera_intrinsics(f, img_name)
                
                # Read camera poses (world-to-camera)
                data['poses_w2c'][img_name] = self._read_camera_pose(f, img_name)

            # Read correspondences only for selected images using proper decoding
            for pair_name in f['corres_i2j'].keys():
                # Parse pair name (e.g., '000000_000001')
                parts = pair_name.split('_')
                src_img = f"{parts[0]}.jpg"
                dst_img = f"{parts[1]}.jpg"
                
                # Only include correspondences where both images are in selected set
                if src_img in selected_images and dst_img in selected_images:
                    coords_src, coords_dst, certainty = self._read_corres(f, pair_name)
                    data['corres_i2j'][pair_name] = {
                        'coords_src': coords_src,
                        'coords_dst': coords_dst,
                        'certainty': certainty
                    }
        
        return data
    
    def _select_random_images(self, image_names: List[str], num_images: int = 5) -> List[str]:
        """Select random images from available images."""
        if len(image_names) <= num_images:
            return image_names
        return random.sample(image_names, num_images)
    
    
    def __len__(self) -> int:
        """Return number of scenes (folders)."""
        return len(self.scene_folders)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item for a single scene.
        
        Args:
            idx: Scene index
            
        Returns:
            Dictionary containing:
            - scene_name: Name of the scene
            - rgb: Dict of selected RGB images as torch tensors (float32)
            - depth_pr: Dict of selected depth predictions as torch tensors (float32)
            - corres_i2j: Dict of correspondences between selected images as torch tensors (float32)
            - src_idx, dst_idx: Tensors of correspondence pair indices (num_pairs,)
            - src_coords, dst_coords: Tensors of sampled correspondence coordinates (num_pairs, K, 2)
            - src_depth, dst_depth: Tensors of sampled depth values (num_pairs, K)
        """
        scene_folder = self.scene_folders[idx]
        scene_name = os.path.basename(scene_folder)
        hdf5_path = self._get_hdf5_path(scene_folder)
        
        # First, get available image names efficiently
        available_images = self._get_available_images(hdf5_path)
        random.shuffle(available_images)
        if len(available_images) == 0:
            raise ValueError(f"No images found in {hdf5_path}")
        
        # Select random images (or all if less than specified number)
        selected_images = self._select_random_images(available_images, num_images=self.num_images)
        
        # Apply natural sorting to ensure good order
        selected_images = natsorted(selected_images)
        
        # Read only selected data from HDF5 for efficiency
        data = self._read_selected_hdf5_data(hdf5_path, selected_images)
        
        # Prepare output data
        result = {
            'scene_name': scene_name,
            'selected_images': selected_images,  # Include selected images list to maintain order
            'rgb': {},
            'rgb_processed': None,  # Will be stacked tensor
            'depth_pr': {},
            'depth_pr_processed': None,  # Will be stacked tensor
            'corres_i2j': {},
            'shared_focal': self.shared_focal
        }
        
        # Lists to collect processed tensors in order
        rgb_processed_list = []
        depth_processed_list = []
        
        # Lists to collect transformation matrices in order
        K_to_K_prime_list = []
        K_prime_to_K_list = []
        K_list = []
        poses_list = []
        
        # Process selected images (RGB and depth) using unified ImagePreprocessor
        for img_name in selected_images:
            # Assertions for data availability
            assert img_name in data['rgb'], f"RGB image {img_name} not found in data"
            assert img_name in data['depth_pr'], f"Depth image {img_name} not found in data"
            assert img_name in data['poses_w2c'], f"Camera pose {img_name} not found in data"
            
            # === RGB PROCESSING ===
            # Keep original RGB as tensor for backward compatibility
            result['rgb'][img_name] = self.image_preprocessor.to_tensor(data['rgb'][img_name], is_depth=False)
            
            # Process RGB image with ImagePreprocessor
            rgb_pil = data['rgb'][img_name]
            rgb_processed_tensor, K_to_K_prime, K_prime_to_K = self.image_preprocessor.process_image_with_matrices(rgb_pil, is_depth=False)
            
            # Collect processed tensor (remove batch dimension for stacking)
            rgb_processed_list.append(rgb_processed_tensor.squeeze(0))
            
            # Collect matrices for stacking (only need to do this once per image since RGB and depth have same transformation)
            K_to_K_prime_list.append(K_to_K_prime)
            K_prime_to_K_list.append(K_prime_to_K)
            K_list.append(data['intrinsics'][img_name])
            
            # === DEPTH PROCESSING ===
            # Process depth image with ImagePreprocessor (handles PIL operations + tensor conversion)
            depth_pil = data['depth_pr'][img_name]
            depth_processed_tensor, _, _ = self.image_preprocessor.process_image_with_matrices(depth_pil, is_depth=True)
            
            # Collect processed tensor (remove batch dimension for stacking)
            depth_processed_list.append(depth_processed_tensor.squeeze(0))
            
            # For backward compatibility, also store original depth as tensor
            # Use the same conversion logic as ImagePreprocessor for consistency
            # Squeeze out channel dimension to match expected shape (H, W) for sample_correspondence_and_depth
            result['depth_pr'][img_name] = self.image_preprocessor.to_tensor(depth_pil, is_depth=True).squeeze(0)
            
            # === POSE PROCESSING ===
            # Store and collect camera poses
            result['poses_w2c'] = data['poses_w2c']  # Store individual poses by name
            poses_list.append(data['poses_w2c'][img_name])
        
        # Stack processed tensors to maintain order
        result['rgb_processed'] = torch.stack(rgb_processed_list, dim=0)  # Shape: [N, C, H, W]
        result['depth_pr_processed'] = torch.stack(depth_processed_list, dim=0)  # Shape: [N, 1, H, W]
        
        # Stack transformation matrices to maintain order
        result['K_to_K_prime'] = torch.stack(K_to_K_prime_list, dim=0)  # Shape: [N, 3, 3]
        result['K_prime_to_K'] = torch.stack(K_prime_to_K_list, dim=0)  # Shape: [N, 3, 3]
        result['K'] = torch.stack(K_list, dim=0)  # Shape: [N, 3, 3]
        result['poses_w2c'] = torch.stack(poses_list, dim=0)  # Shape: [N, 4, 4] or [N, 3, 4]
        
        # Process correspondences
        for pair_name, corr_data in data['corres_i2j'].items():
            result['corres_i2j'][pair_name] = {
                'coords_src': torch.from_numpy(corr_data['coords_src'].astype(np.float32)).to(torch.float32),
                'coords_dst': torch.from_numpy(corr_data['coords_dst'].astype(np.float32)).to(torch.float32),
                'certainty': torch.from_numpy(corr_data['certainty'].astype(np.float32)).to(torch.float32)
            }
        
        # Create downsampled correspondence arrays
        self._create_downsampled_arrays(result, selected_images)
        
        return result
    
    def _create_downsampled_arrays(self, result: Dict[str, Any], selected_images: List[str]):
        """
        Create downsampled correspondence arrays for efficient batch processing.
        Only uses processed data from result dictionary.
        
        Args:
            result: Result dictionary to populate with arrays
            selected_images: List of selected image names
        """
        # Create image name to index mapping
        img_to_idx = {img_name: idx for idx, img_name in enumerate(selected_images)}

        # Collect all valid correspondence pairs using processed data from result
        valid_pairs = []
        for pair_name, corr_data in result['corres_i2j'].items():
            parts = pair_name.split('_')
            src_img = f"{parts[0]}.jpg"
            dst_img = f"{parts[1]}.jpg"
            
            # Include all pairs where both images are in selected set
            assert src_img in img_to_idx and dst_img in img_to_idx, f"Images {src_img} or {dst_img} not in selected images"
            src_image_idx = img_to_idx[src_img]
            dst_image_idx = img_to_idx[dst_img]
            valid_pairs.append((src_image_idx, dst_image_idx, pair_name, corr_data))

        # Initialize arrays
        src_indices = []
        dst_indices = []
        src_depths_list = []
        dst_depths_list = []
        src_coords_list = []
        dst_coords_list = []
        
        # Process each correspondence pair
        for src_image_idx, dst_image_idx, pair_name, corr_data in valid_pairs:
            # Get image names
            src_img = selected_images[src_image_idx]
            dst_img = selected_images[dst_image_idx]
            
            # Get depth maps from processed result (already converted to meters)
            depth_src = result['depth_pr'][src_img].numpy()  # Already in meters
            depth_dst = result['depth_pr'][dst_img].numpy()  # Already in meters
            
            # Convert torch tensors to numpy arrays for sampling function
            coords_src_np = corr_data['coords_src'].numpy()
            coords_dst_np = corr_data['coords_dst'].numpy()
            certainty_np = corr_data['certainty'].numpy()
            
            # Sample correspondences and depths
            sampled_src_coords, sampled_dst_coords, sampled_src_depth, sampled_dst_depth = sample_correspondence_and_depth(
                coords_src=coords_src_np,
                coords_dst=coords_dst_np,
                certainty=certainty_np,
                depth_src=depth_src,
                depth_dst=depth_dst,
                sample_num=self.sample_num,
                min_corres_conf=self.min_corres_conf
            )
            
            # Store results
            src_indices.append(src_image_idx)
            dst_indices.append(dst_image_idx)
            src_coords_list.append(sampled_src_coords)
            dst_coords_list.append(sampled_dst_coords)
            src_depths_list.append(sampled_src_depth)
            dst_depths_list.append(sampled_dst_depth)
        
        # Convert to tensors with expected shapes
        result['src_idx'] = torch.tensor(src_indices, dtype=torch.long)  # (num_pairs,)
        result['dst_idx'] = torch.tensor(dst_indices, dtype=torch.long)  # (num_pairs,)
        result['src_coords'] = torch.from_numpy(np.stack(src_coords_list, axis=0)).to(torch.float32)  # (num_pairs, K, 2)
        result['dst_coords'] = torch.from_numpy(np.stack(dst_coords_list, axis=0)).to(torch.float32)  # (num_pairs, K, 2)
        result['src_depth'] = torch.from_numpy(np.stack(src_depths_list, axis=0)).to(torch.float32)  # (num_pairs, K)
        result['dst_depth'] = torch.from_numpy(np.stack(dst_depths_list, axis=0)).to(torch.float32)  # (num_pairs, K)

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for handling variable number of correspondences
    across different GPUs in multi-GPU training.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched data with proper handling of variable correspondences
    """
    # Assert that batch size must be 1
    assert len(batch) == 1, f"Batch size must be 1, got {len(batch)}"
    
    # Single scene per GPU - just return the scene data
    return batch[0]


# Example usage:
if __name__ == "__main__":
    # Test the dataset
    dataset = IMC2021(root="/home/ubuntu/disk6/Motion-from-Structure/release/imc2021/DUSt3R_RoMa")
    print(f"Dataset length: {len(dataset)}")

    # Test single item
    sample = dataset[0]
    
    # Test with DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # One scene per GPU
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging, increase for performance
    )

    for batch in dataloader:
        print(f"Batch scene: {batch['scene_name']}")
        print(f"Batch correspondences: {len(batch['src_idx'])}")
        
        # Import visualization functions
        import sys
        import os
        
        # Add the parent directory (train) to the path
        train_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, train_dir)
        
        # Import from utils directory relative to train
        from utils.vls import monodepth2vls, image2vls, corres2vls
        
        # Visualize one RGB/depth pair (random selection from processed tensors)
        selected_images = batch['selected_images']
        random_img_idx = random.randint(0, len(selected_images) - 1)
        random_img = selected_images[random_img_idx]
        
        # Get original RGB tensor for comparison (add batch dimension)
        rgb_tensor = batch['rgb'][random_img].unsqueeze(0)
        
        # Get processed tensors (already have correct dimensions)
        rgb_processed_tensor = batch['rgb_processed'][random_img_idx].unsqueeze(0)  # Shape: [1, C, H, W]
        depth_processed_tensor = batch['depth_pr_processed'][random_img_idx]  # Shape: [1, H, W]
        
        # Get original depth tensor for comparison (add batch and channel dimensions)
        depth_tensor = batch['depth_pr'][random_img].unsqueeze(0).unsqueeze(0)

        rgb_img = image2vls(rgb_tensor)
        rgb_processed_img = image2vls(rgb_processed_tensor)
        depth_img = monodepth2vls(1 / depth_tensor, vmax=2.0)
        depth_processed_img = monodepth2vls(1 / depth_processed_tensor, vmax=2.0)

        # Create side-by-side RGB and depth visualization (unprocessed)
        rgb_array = np.array(rgb_img)
        depth_array = np.array(depth_img)
        
        # Concatenate horizontally (side by side)
        combined_array = np.concatenate([rgb_array, depth_array], axis=1)
        combined_img = PIL.Image.fromarray(combined_array)

        # Create side-by-side processed RGB and depth visualization
        rgb_processed_array = np.array(rgb_processed_img)
        depth_processed_array = np.array(depth_processed_img)
        
        # Concatenate horizontally (side by side)
        combined_processed_array = np.concatenate([rgb_processed_array, depth_processed_array], axis=1)
        combined_processed_img = PIL.Image.fromarray(combined_processed_array)

        # Save combined RGB-depth visualization (unprocessed)
        combined_img.save(f"/home/ubuntu/tmp/rgb_depth_{batch['scene_name']}_{random_img}.png")
        
        # Save combined RGB-depth visualization (processed)
        combined_processed_img.save(f"/home/ubuntu/tmp/rgb_depth_processed_{batch['scene_name']}_{random_img}.png")
        
        # Print image sizes and intrinsic matrices for the selected image
        print(f"\n=== Image and Intrinsics Analysis for {random_img} ===")
        print(f"Original RGB image size: {rgb_tensor.shape}")  # [1, C, H, W]
        print(f"Processed RGB image size: {rgb_processed_tensor.shape}")  # [1, C, H, W]
        print(f"Original depth image size: {depth_tensor.shape}")  # [1, 1, H, W]
        print(f"Processed depth image size: {depth_processed_tensor.shape}")  # [1, H, W]
        
        # Get original intrinsics and transformation matrices (using integer index for stacked tensors)
        K_original = batch['K'][random_img_idx]
        K_prime_to_K = batch['K_prime_to_K'][random_img_idx]
        
        print(f"\nOriginal intrinsic matrix K:")
        print(K_original)
        print(f"\nReverse transformation matrix (K_prime_to_K):")
        print(K_prime_to_K)
        
        # Demonstrate the reverse transformation
        K_to_K_prime = batch['K_to_K_prime'][random_img_idx]
        K_prime_computed = K_to_K_prime @ K_original
        K_recovered = K_prime_to_K @ K_prime_computed
        
        print(f"\nTransformed intrinsics K_prime (K_to_K_prime @ K):")
        print(K_prime_computed)
        print(f"\nRecovered original K (K_prime_to_K @ K_prime):")
        print(K_recovered)
        print(f"\nRecovery error (should be close to zero):")
        print(torch.abs(K_original - K_recovered).max().item())

        # Visualize one correspondence pair
        import random
        pair_name = random.choice(list(batch['corres_i2j'].keys()))
        parts = pair_name.split('_')

        src_img = f"{parts[0]}.jpg"
        dst_img = f"{parts[1]}.jpg"

        # Get image indices for the selected images
        selected_images = batch['selected_images']
        src_img_idx = selected_images.index(src_img)
        dst_img_idx = selected_images.index(dst_img)
        
        # Get RGB tensors for correspondence visualization (use original tensors since correspondences are in original space)
        rgb_src_tensor = batch['rgb'][src_img].unsqueeze(0)  # Shape: [1, C, H, W]
        rgb_dst_tensor = batch['rgb'][dst_img].unsqueeze(0)  # Shape: [1, C, H, W]

        corr_data = batch['corres_i2j'][pair_name]
        coords_src = corr_data['coords_src'].unsqueeze(0)
        coords_dst = corr_data['coords_dst'].unsqueeze(0)
        certainty = corr_data['certainty'].unsqueeze(0).unsqueeze(0)

        corres_combined = torch.cat([coords_src, coords_dst], dim=-1)
        corres_img = corres2vls(rgb_src_tensor, rgb_dst_tensor, corres_combined, certainty)

        # Save correspondence visualization to /home/ubuntu/tmp
        corres_img.save(f"/home/ubuntu/tmp/corres_{batch['scene_name']}_{pair_name}.png")

        # Add matplotlib visualization for sampled correspondences and depth maps
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Assume we have sampled correspondence arrays (len(batch['src_idx']) > 0)
        # Randomly select a correspondence pair from the sampled arrays
        random_pair_idx = random.randint(0, len(batch['src_idx']) - 1)
        
        src_img_idx = batch['src_idx'][random_pair_idx].item()
        dst_img_idx = batch['dst_idx'][random_pair_idx].item()
        
        # Get image names from selected_images list (maintains order)
        selected_images = batch['selected_images']
        src_img_name = selected_images[src_img_idx]
        dst_img_name = selected_images[dst_img_idx]
        
        # Get sampled correspondence coordinates and depths
        sampled_src_coords = batch['src_coords'][random_pair_idx].numpy()  # (K, 2)
        sampled_dst_coords = batch['dst_coords'][random_pair_idx].numpy()  # (K, 2)
        sampled_src_depth = batch['src_depth'][random_pair_idx].numpy()    # (K,)
        sampled_dst_depth = batch['dst_depth'][random_pair_idx].numpy()    # (K,)
        
        # Limit to 10 points for visualization
        num_vis_points = min(10, len(sampled_src_coords))
        vis_indices = np.random.choice(len(sampled_src_coords), num_vis_points, replace=False)
        sampled_src_coords = sampled_src_coords[vis_indices]
        sampled_dst_coords = sampled_dst_coords[vis_indices]
        sampled_src_depth = sampled_src_depth[vis_indices]
        sampled_dst_depth = sampled_dst_depth[vis_indices]
        
        # Get RGB images as numpy arrays for visualization (use original images since correspondences are in original space)
        src_rgb = batch['rgb'][src_img_name].permute(1, 2, 0).numpy()  # (H, W, 3)
        dst_rgb = batch['rgb'][dst_img_name].permute(1, 2, 0).numpy()  # (H, W, 3)

        # Get full depth maps (use original depth maps since correspondences are in original space)
        src_depth_full = batch['depth_pr'][src_img_name].numpy()  # (H, W)
        dst_depth_full = batch['depth_pr'][dst_img_name].numpy()  # (H, W)

        # Use coordinates directly (they are already in image coordinate system)
        H, W = src_rgb.shape[:2]
        src_coords_pixel = sampled_src_coords
        dst_coords_pixel = sampled_dst_coords
        
        # Create matplotlib figure with 3 rows and 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        
        # Define colormap for depth visualization
        depth_colormap = cm.viridis
        
        # Combine all depth values to get consistent color scale
        all_depths = np.concatenate([sampled_src_depth, sampled_dst_depth, 
                                   src_depth_full.flatten(), dst_depth_full.flatten()])
        # Filter out zero/invalid depths for better visualization (assume len(valid_depths) > 0)
        valid_depths = all_depths[all_depths > 0]
        vmin, vmax = np.percentile(valid_depths, [5, 95])  # Use 5th-95th percentile for better contrast
        
        # Row 1: RGB images with correspondence points
        axes[0, 0].imshow(src_rgb)
        axes[0, 0].scatter(src_coords_pixel[:, 0], src_coords_pixel[:, 1], 
                         c='red', s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
        axes[0, 0].set_title(f'Source Image: {src_img_name}\nCorrespondence Points (Red)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(dst_rgb)
        axes[0, 1].scatter(dst_coords_pixel[:, 0], dst_coords_pixel[:, 1], 
                         c='red', s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
        axes[0, 1].set_title(f'Destination Image: {dst_img_name}\nCorrespondence Points (Red)')
        axes[0, 1].axis('off')
        
        # Row 2: RGB images with scattered depth points overlay (to verify sampling accuracy)
        # Show RGB images as background with sampled depth points overlaid
        
        # Display RGB images as background
        axes[1, 0].imshow(src_rgb)
        scatter_src = axes[1, 0].scatter(src_coords_pixel[:, 0], src_coords_pixel[:, 1], 
                                       c=sampled_src_depth, s=60, cmap=depth_colormap, 
                                       vmin=vmin, vmax=vmax, alpha=1.0, edgecolors='white', linewidth=1.0)
        axes[1, 0].set_title(f'RGB + Sampled Depth Points - Source\n({len(sampled_src_depth)} points overlaid)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(dst_rgb)
        scatter_dst = axes[1, 1].scatter(dst_coords_pixel[:, 0], dst_coords_pixel[:, 1], 
                                       c=sampled_dst_depth, s=60, cmap=depth_colormap, 
                                       vmin=vmin, vmax=vmax, alpha=1.0, edgecolors='white', linewidth=1.0)
        axes[1, 1].set_title(f'RGB + Sampled Depth Points - Destination\n({len(sampled_dst_depth)} points overlaid)')
        axes[1, 1].axis('off')
        
        # Add colorbar for depth points (use shrink to avoid tight_layout issues)
        # Use the scatter plot for colorbar to maintain same scale as third row
        cbar1 = plt.colorbar(scatter_dst, ax=axes[1, 1], shrink=0.8)
        cbar1.set_label('Depth (meters)', fontsize=10)
        
        # Row 3: Full depth maps
        # Mask invalid depths (zeros) for better visualization
        src_depth_masked = np.where(src_depth_full > 0, src_depth_full, np.nan)
        dst_depth_masked = np.where(dst_depth_full > 0, dst_depth_full, np.nan)
        
        im_src = axes[2, 0].imshow(src_depth_masked, cmap=depth_colormap, vmin=vmin, vmax=vmax)
        axes[2, 0].scatter(src_coords_pixel[:, 0], src_coords_pixel[:, 1], 
                         c='red', s=15, alpha=0.8, edgecolors='white', linewidth=0.5)
        axes[2, 0].set_title(f'Full Depth Map - Source\n(Red dots: {num_vis_points} sampled points)')
        axes[2, 0].axis('off')
        
        im_dst = axes[2, 1].imshow(dst_depth_masked, cmap=depth_colormap, vmin=vmin, vmax=vmax)
        axes[2, 1].scatter(dst_coords_pixel[:, 0], dst_coords_pixel[:, 1], 
                         c='red', s=15, alpha=0.8, edgecolors='white', linewidth=0.5)
        axes[2, 1].set_title(f'Full Depth Map - Destination\n(Red dots: {num_vis_points} sampled points)')
        axes[2, 1].axis('off')
        
        # Add colorbar for full depth maps (use shrink to avoid tight_layout issues)
        cbar2 = plt.colorbar(im_dst, ax=axes[2, 1], shrink=0.8)
        cbar2.set_label('Depth (meters)', fontsize=10)
        
        # Add overall title
        fig.suptitle(f'Correspondence and Depth Visualization\nScene: {batch["scene_name"]}\n'
                    f'Pair: {src_img_name} ↔ {dst_img_name} ({num_vis_points} points)\n'
                    f'Row 2: RGB images with overlaid sampled depth points for verification', fontsize=14)
        
        # Use subplots_adjust instead of tight_layout to avoid warnings
        plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.2)
        
        # Save the comprehensive visualization
        plt.savefig(f"/home/ubuntu/tmp/correspondence_depth_analysis_{batch['scene_name']}_pair_{random_pair_idx}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print detailed comparison of sampled depth values vs depth map pixel values
        print(f"\n=== Correspondence and Depth Analysis ===")
        print(f"Visualization saved to: /home/ubuntu/tmp/correspondence_depth_analysis_{batch['scene_name']}_pair_{random_pair_idx}.png")
        
        # Round coordinates to integers for pixel lookup
        src_coords_int = np.round(src_coords_pixel).astype(int)
        dst_coords_int = np.round(dst_coords_pixel).astype(int)
        
        # Ensure coordinates are within image bounds
        src_coords_int[:, 0] = np.clip(src_coords_int[:, 0], 0, W-1)
        src_coords_int[:, 1] = np.clip(src_coords_int[:, 1], 0, H-1)
        dst_coords_int[:, 0] = np.clip(dst_coords_int[:, 0], 0, W-1)
        dst_coords_int[:, 1] = np.clip(dst_coords_int[:, 1], 0, H-1)
        
        # Get depth map values at rounded pixel locations
        src_depth_at_pixels = src_depth_full[src_coords_int[:, 1], src_coords_int[:, 0]]
        dst_depth_at_pixels = dst_depth_full[dst_coords_int[:, 1], dst_coords_int[:, 0]]
        
        print(f"\n=== Sampled Depth vs Depth Map Pixel Comparison ===")
        print(f"Source Image ({src_img_name}):")
        for i in range(num_vis_points):
            coord_exact = src_coords_pixel[i]
            coord_int = src_coords_int[i]
            sampled_val = sampled_src_depth[i]
            pixel_val = src_depth_at_pixels[i]
            diff = abs(sampled_val - pixel_val)
            print(f"  Point {i+1}: Coord({coord_exact[0]:.2f}, {coord_exact[1]:.2f}) -> Pixel({coord_int[0]}, {coord_int[1]})")
            print(f"           Sampled: {sampled_val:.4f}m, Pixel: {pixel_val:.4f}m, Diff: {diff:.4f}m")
        
        print(f"\nDestination Image ({dst_img_name}):")
        for i in range(num_vis_points):
            coord_exact = dst_coords_pixel[i]
            coord_int = dst_coords_int[i]
            sampled_val = sampled_dst_depth[i]
            pixel_val = dst_depth_at_pixels[i]
            diff = abs(sampled_val - pixel_val)
            print(f"  Point {i+1}: Coord({coord_exact[0]:.2f}, {coord_exact[1]:.2f}) -> Pixel({coord_int[0]}, {coord_int[1]})")
            print(f"           Sampled: {sampled_val:.4f}m, Pixel: {pixel_val:.4f}m, Diff: {diff:.4f}m")
        
        # Calculate and print summary statistics
        src_diffs = np.abs(sampled_src_depth - src_depth_at_pixels)
        dst_diffs = np.abs(sampled_dst_depth - dst_depth_at_pixels)
        all_diffs = np.concatenate([src_diffs, dst_diffs])
        
        print(f"\n=== Summary Statistics ===")
        print(f"Source depth differences - Mean: {np.mean(src_diffs):.4f}m, Max: {np.max(src_diffs):.4f}m")
        print(f"Destination depth differences - Mean: {np.mean(dst_diffs):.4f}m, Max: {np.max(dst_diffs):.4f}m")
        print(f"Overall differences - Mean: {np.mean(all_diffs):.4f}m, Max: {np.max(all_diffs):.4f}m")
        print(f"Note: Differences are expected due to bilinear interpolation vs nearest neighbor pixel lookup")

        break