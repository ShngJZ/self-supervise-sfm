import torch
import numpy as np
import cv2
import copy
import tqdm
from PIL import Image
from torchvision import transforms as TF
from typing import Optional, Tuple, Dict, Any, Union, List

class ImagePreprocessor:
    """
    A class for loading and preprocessing a single image with support for forward and reverse operations
    on camera intrinsics. This class is designed to be compatible with PyTorch DataLoader.
    
    The class preserves transformation parameters to enable recovery of original camera
    intrinsics from modified intrinsics after image preprocessing.
    
    Processing logic:
    1. Pad to the size of the largest side (height or width) using zero padding
    2. Resize to the target size
    
    Supports both RGB images and depth images (uint16 PNG).
    """
    
    def __init__(self, target_size: int = 518):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            target_size (int): Target size for preprocessing (default: 518)
        """
        self.target_size = target_size
        self.default_to_tensor = TF.ToTensor()
    
    def __call__(self, image: Image.Image, is_depth: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single PIL Image with transformation matrices for intrinsics.
        
        Args:
            image (Image.Image): PIL Image to process
            is_depth (bool): Whether this is a depth image (True) or RGB image (False)
        
        Returns:
            Tuple containing:
            - torch.Tensor: Preprocessed image with shape (1, 3, H, W) for RGB or (1, 1, H, W) for depth
            - torch.Tensor: Transformation matrix K_to_K_prime (3, 3) - transforms original K to processed K'
            - torch.Tensor: Transformation matrix K_prime_to_K (3, 3) - transforms processed K' back to original K
        """
        return self.process_image_with_matrices(image, is_depth)
    
    def to_tensor(self, image: Image.Image, is_depth: bool = False) -> torch.Tensor:
        """
        Convert PIL Image to tensor with appropriate handling for RGB and depth images.
        
        Args:
            image (Image.Image): PIL Image to convert
            is_depth (bool): Whether this is a depth image (True) or RGB image (False)
            
        Returns:
            torch.Tensor: Converted tensor
        """
        if is_depth:
            # Handle depth image - convert uint16 to float32 without normalization
            monodepth_uint16 = np.array(image)
            monodepth_uint16 = monodepth_uint16.astype(np.float32)
            monodepth = monodepth_uint16 / 1000  # Convert from mm to meters
            
            # Convert to tensor and add channel dimension: (H, W) -> (1, H, W)
            return torch.from_numpy(monodepth).unsqueeze(0)
        else:
            # Handle RGB image - use default torchvision behavior
            return self.default_to_tensor(image)
    
    
    def process_image_with_matrices(self, image: Image.Image, is_depth: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single PIL Image and return transformation matrices for intrinsics.
        
        Unified processing logic for both RGB and depth images:
        1. Pad to the size of the largest side (height or width) using zero padding
        2. Resize to the target size using bicubic resampling
        3. Convert to tensor with appropriate handling for RGB vs depth
        
        Args:
            image (Image.Image): PIL Image to process
            is_depth (bool): Whether this is a depth image (True) or RGB image (False)
        
        Returns:
            Tuple containing:
            - torch.Tensor: Preprocessed image with shape (1, 3, H, W) for RGB or (1, 1, H, W) for depth
            - torch.Tensor: Transformation matrix K_to_K_prime (3, 3) - transforms original K to processed K'
            - torch.Tensor: Transformation matrix K_prime_to_K (3, 3) - transforms processed K' back to original K
        """
        # Handle image conversion based on type
        if is_depth:
            # For depth images, keep original format (uint16 PIL image)
            img = image
        else:
            # Convert to RGB for RGB images (assumes no RGBA handling needed)
            img = image.convert("RGB")
        
        original_width, original_height = img.size
        
        # Step 1: Pad to the size of the largest side using zero padding
        max_side = max(original_width, original_height)
        
        # Calculate padding needed
        pad_width = max_side - original_width
        pad_height = max_side - original_height
        
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        
        # UNIFIED PIL PROCESSING: Apply zero padding in PIL format for both RGB and depth
        if pad_width > 0 or pad_height > 0:
            # Create new image with padding (works for both RGB and uint16 depth images)
            padded_img = Image.new(img.mode, (max_side, max_side), color=0)
            padded_img.paste(img, (pad_left, pad_top))
            img = padded_img
        
        # UNIFIED PIL PROCESSING: Resize to target size (works for both RGB and uint16 depth images)
        img = img.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)
        
        # Convert to tensor with appropriate handling for RGB vs depth
        img_tensor = self.to_tensor(img, is_depth)
        
        # Calculate transformation parameters
        resize_scale = self.target_size / max_side
        
        transform_params = {
            'scale_x': resize_scale,
            'scale_y': resize_scale,
            'offset_x': pad_left * resize_scale,
            'offset_y': pad_top * resize_scale,
        }
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, 3, H, W) or (1, 1, H, W)
        
        # Create transformation matrices
        K_to_K_prime, K_prime_to_K = self._create_transformation_matrices(transform_params)
        
        return img_tensor, K_to_K_prime, K_prime_to_K
    
    def _create_transformation_matrices(self, transform_param: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create transformation matrices for converting between original and processed intrinsics.
        
        Image processing pipeline:
        1. Original image (W×H) -> Pad to square (max_side×max_side) -> Resize to (target_size×target_size)
        2. Point transformation: (x,y) -> (x+pad_left, y+pad_top) -> ((x+pad_left)*scale, (y+pad_top)*scale)
        3. Overall: x' = x*scale + pad_left*scale, y' = y*scale + pad_top*scale
        
        Camera intrinsic transformation:
        - fx' = fx * scale, fy' = fy * scale (focal lengths scale with image)
        - cx' = cx * scale + pad_left * scale (principal point shifts and scales)
        - cy' = cy * scale + pad_top * scale
        
        Args:
            transform_param (Dict): Transformation parameters containing scale and offset values
        
        Returns:
            Tuple containing:
            - torch.Tensor: K_to_K_prime matrix (3, 3) - transforms original K to processed K'
            - torch.Tensor: K_prime_to_K matrix (3, 3) - transforms processed K' back to original K
        
        Usage:
            K_prime = K_to_K_prime @ K_original  
            K_recovered = K_prime_to_K @ K_prime  # Should equal K_original
        """
        # Extract transformation parameters
        scale_x = transform_param['scale_x']  # resize_scale (target_size / max_side)
        scale_y = transform_param['scale_y']  # resize_scale (same as scale_x for square)
        offset_x = transform_param['offset_x']  # pad_left * resize_scale
        offset_y = transform_param['offset_y']  # pad_top * resize_scale
        
        # Forward transformation matrix: K_original -> K_processed
        # Applies the combined padding + resizing transformation to intrinsic parameters
        K_to_K_prime = torch.tensor([
            [scale_x, 0.0, offset_x],
            [0.0, scale_y, offset_y], 
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        # Reverse transformation matrix: K_processed -> K_original  
        # Mathematically inverts the forward transformation
        K_prime_to_K = torch.tensor([
            [1.0/scale_x, 0.0, -offset_x/scale_x],
            [0.0, 1.0/scale_y, -offset_y/scale_y],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        return K_to_K_prime, K_prime_to_K

    def reverse_transform_tensor(self, processed_tensor: torch.Tensor, K_prime_to_K: torch.Tensor, 
                                target_size: int, is_depth: bool = False) -> torch.Tensor:
        """
        Reverse the preprocessing transformations applied to a tensor to get back to original dimensions.
        
        This method reverses the padding and resizing operations to restore the tensor to its 
        original image dimensions using the inverse transformation matrix.
        
        Args:
            processed_tensor (torch.Tensor): Processed tensor with shape (C, target_size, target_size)
                                           where C=3 for RGB or C=1 for depth
            K_prime_to_K (torch.Tensor): Inverse transformation matrix (3, 3) that transforms 
                                        processed intrinsics back to original intrinsics
            target_size (int): The target size used during preprocessing (e.g., 518)
            is_depth (bool): Whether this is a depth tensor (True) or RGB tensor (False)
            
        Returns:
            torch.Tensor: Tensor with original image dimensions, shape (C, original_height, original_width)
        """
        # Extract transformation parameters from the K_prime_to_K matrix
        # K_prime_to_K has the form:
        # [[1.0/scale_x, 0.0, -offset_x/scale_x],
        #  [0.0, 1.0/scale_y, -offset_y/scale_y],
        #  [0.0, 0.0, 1.0]]
        
        scale_x = 1.0 / K_prime_to_K[0, 0].item()  # resize_scale (target_size / max_side)
        scale_y = 1.0 / K_prime_to_K[1, 1].item()  # resize_scale (same as scale_x)
        offset_x = -K_prime_to_K[0, 2].item() * scale_x  # pad_left * resize_scale  
        offset_y = -K_prime_to_K[1, 2].item() * scale_y  # pad_top * resize_scale
        
        # Calculate reverse transformation parameters
        # Step 1: Calculate original max_side from the scale
        max_side = int(target_size / scale_x)
        
        # Calculate original padding (before scaling)
        pad_left = int(offset_x / scale_x)
        pad_top = int(offset_y / scale_y)
        
        # Step 2: Resize from target_size back to max_side
        # processed_tensor shape: (C, target_size, target_size)
        C = processed_tensor.shape[0]
        
        # Use bicubic interpolation for upsampling (reverse of the downsampling during processing)
        resized_tensor = torch.nn.functional.interpolate(
            processed_tensor.unsqueeze(0),  # Add batch dim: (1, C, target_size, target_size)
            size=(max_side, max_side),
            mode='bicubic' if not is_depth else 'bilinear',  # Use bilinear for depth to avoid artifacts
            align_corners=False
        ).squeeze(0)  # Remove batch dim: (C, max_side, max_side)
        
        # Step 3: Remove padding to get back to original dimensions  
        # Calculate original dimensions from padding
        original_width = max_side - 2 * pad_left  # Remove left and right padding
        original_height = max_side - 2 * pad_top  # Remove top and bottom padding
        
        # Crop out the padded regions
        original_tensor = resized_tensor[
            :, 
            pad_top:pad_top + original_height,  # Remove top and bottom padding
            pad_left:pad_left + original_width   # Remove left and right padding
        ]
        
        return original_tensor


def torchncoords2coordinates(coords: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Convert normalized coordinates [-1, 1] to pixel coordinates [0, w-1] and [0, h-1].
    
    Args:
        coords (np.ndarray): Normalized coordinates with shape (..., 2)
        h (int): Image height
        w (int): Image width
        
    Returns:
        np.ndarray: Pixel coordinates with same shape as input
    """
    coords_pixel = coords.copy()
    coords_pixel[..., 0] = (coords[..., 0] + 1) * (w - 1) / 2  # x coordinates
    coords_pixel[..., 1] = (coords[..., 1] + 1) * (h - 1) / 2  # y coordinates
    return coords_pixel


def sample_correspondence_and_depth(
    coords_src: np.ndarray,
    coords_dst: np.ndarray,
    certainty: np.ndarray,
    depth_src: np.ndarray,
    depth_dst: np.ndarray,
    sample_num: int,
    min_corres_conf: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample correspondence points and their associated depth values from dense maps.
    
    This function takes correspondence and depth data in a single direction and samples 
    a fixed number of correspondence points based on confidence weights, then extracts 
    depth values at those sampled locations using bilinear interpolation.
    
    Args:
        coords_src (np.ndarray): Source coordinates in normalized format [-1, 1] with shape (N, 2)
        coords_dst (np.ndarray): Destination coordinates in normalized format [-1, 1] with shape (N, 2)
        certainty (np.ndarray): Confidence/certainty values for correspondences with shape (N,)
        depth_src (np.ndarray): Source depth map with shape (H1, W1)
        depth_dst (np.ndarray): Destination depth map with shape (H2, W2)
        sample_num (int): Number of correspondence points to sample
        min_corres_conf (float): Minimum confidence threshold for filtering (default: 0.0)
        
    Returns:
        Tuple containing:
        - coords_src_sampled (np.ndarray): Sampled source coordinates in pixel format with shape (sample_num, 2)
        - coords_dst_sampled (np.ndarray): Sampled destination coordinates in pixel format with shape (sample_num, 2)
        - depth_src_sampled (np.ndarray): Depth values at sampled source locations with shape (sample_num,)
        - depth_dst_sampled (np.ndarray): Depth values at sampled destination locations with shape (sample_num,)
    """
    
    # Filter correspondences by minimum confidence threshold
    selector = certainty > min_corres_conf
    coords_src_filtered = coords_src[selector, :]
    coords_dst_filtered = coords_dst[selector, :]
    certainty_filtered = certainty[selector]
    
    if len(certainty_filtered) == 0:
        raise ValueError(f"No correspondence points meet the minimum confidence threshold {min_corres_conf}. "
                        f"Total points: {len(certainty)}, points above threshold: 0. "
                        f"Consider lowering min_corres_conf or checking correspondence quality.")

    # Sample correspondence points based on confidence weights
    probabilities = certainty_filtered / np.sum(certainty_filtered)
    sampled_indices = np.random.choice(
        np.arange(len(certainty_filtered)),
        size=sample_num,
        replace=True,
        p=probabilities
    )
    
    coords_src_sampled_norm = coords_src_filtered[sampled_indices, :]
    coords_dst_sampled_norm = coords_dst_filtered[sampled_indices, :]
    
    # Sample depth values at correspondence locations using bilinear interpolation
    h1, w1 = depth_src.shape
    h2, w2 = depth_dst.shape
    
    # Sample depth from source image
    depth_src_sampled = torch.nn.functional.grid_sample(
        torch.from_numpy(depth_src).view(1, 1, h1, w1).float(),
        torch.from_numpy(coords_src_sampled_norm).view(1, 1, sample_num, 2).float(),
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # Sample depth from destination image
    depth_dst_sampled = torch.nn.functional.grid_sample(
        torch.from_numpy(depth_dst).view(1, 1, h2, w2).float(),
        torch.from_numpy(coords_dst_sampled_norm).view(1, 1, sample_num, 2).float(),
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # Convert normalized coordinates to pixel coordinates
    coords_src_sampled = torchncoords2coordinates(coords_src_sampled_norm, h1, w1)
    coords_dst_sampled = torchncoords2coordinates(coords_dst_sampled_norm, h2, w2)
    
    return coords_src_sampled, coords_dst_sampled, depth_src_sampled, depth_dst_sampled


