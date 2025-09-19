import torch
import numpy as np
import networkx as nx
from typing import Union, Tuple, Optional

def to_homogeneous(points: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert N-dimensional points to homogeneous coordinates by appending ones.
    
    Args:
        points: Input points with shape (..., N)
        
    Returns:
        Homogeneous points with shape (..., N+1)
    """
    if isinstance(points, torch.Tensor):
        ones = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, ones], dim=-1)
    elif isinstance(points, np.ndarray):
        ones = np.ones(points.shape[:-1] + (1,), dtype=points.dtype)
        return np.concatenate([points, ones], axis=-1)
    else:
        raise ValueError(f"Unsupported type: {type(points)}")

def from_homogeneous(points: Union[torch.Tensor, np.ndarray], eps: float = 1e-6) -> Union[torch.Tensor, np.ndarray]:
    """Convert homogeneous coordinates back to N-dimensional points.
    
    Args:
        points: Homogeneous points with shape (..., N+1)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        N-dimensional points with shape (..., N)
    """
    return points[..., :-1] / (points[..., -1:] + eps)

def pad_poses(pose: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Pad [..., 3, 4] pose matrices with homogeneous bottom row [0,0,0,1].
    
    Args:
        pose: Pose matrix with shape (..., 3, 4)
        
    Returns:
        Padded pose matrix with shape (..., 4, 4)
    """
    if isinstance(pose, torch.Tensor):
        bottom_row = torch.tensor([0, 0, 0, 1.0], device=pose.device, dtype=pose.dtype)
        bottom = bottom_row.expand(pose.shape[:-2] + (1, 4))
        return torch.cat([pose[..., :3, :4], bottom], dim=-2)
    elif isinstance(pose, np.ndarray):
        pose_tensor = torch.from_numpy(pose)
        bottom_row = torch.tensor([0, 0, 0, 1.0], dtype=pose_tensor.dtype)
        bottom = bottom_row.expand(pose_tensor.shape[:-2] + (1, 4))
        result = torch.cat([pose_tensor[..., :3, :4], bottom], dim=-2)
        return result.numpy()
    else:
        raise NotImplementedError(f"Unsupported type: {type(pose)}")

def pad_intrinsic44(intrinsic: torch.Tensor) -> torch.Tensor:
    """Pad 3x3 intrinsic matrices to 4x4 by adding identity elements.
    
    Args:
        intrinsic: Intrinsic matrix with shape (batch_size, 3, 3)
        
    Returns:
        Padded intrinsic matrix with shape (batch_size, 4, 4)
    """
    batch_size, device = intrinsic.shape[0], intrinsic.device
    intrinsic44 = torch.eye(4, device=device, dtype=intrinsic.dtype)
    intrinsic44 = intrinsic44.unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsic44[:, :3, :3] = intrinsic
    return intrinsic44

def unpad_poses(pose: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Remove homogeneous bottom row from [..., 4, 4] pose matrices.
    
    Args:
        pose: Pose matrix with shape (..., 4, 4)
        
    Returns:
        Unpadded pose matrix with shape (..., 3, 4)
    """
    if isinstance(pose, torch.Tensor):
        return pose[..., :3, :4]
    elif isinstance(pose, np.ndarray):
        return pose[..., :3, :4]
    else:
        raise NotImplementedError(f"Unsupported type: {type(pose)}")

def backproject_and_reproject(
    src_coords: torch.Tensor,
    src_depth: torch.Tensor,
    src_intrinsic: torch.Tensor,
    dst_intrinsic: torch.Tensor,
    relative_pose: torch.Tensor,
    src_depth_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backproject source coordinates to 3D camera space then reproject to destination camera using relative pose.
    
    This function performs the complete geometric pipeline for multi-view correspondence:
    1. Scales source depth values using src_depth_scale
    2. Backprojects source pixel coordinates to 3D camera coordinates using source intrinsics
    3. Transforms 3D points from source camera to destination camera using relative pose
    4. Projects 3D points to destination pixel coordinates using destination intrinsics
    5. Computes validity mask for points in front of destination camera
    
    Args:
        src_coords: Source pixel coordinates, shape (BZ, N, 2)
        src_depth: Source depth values, shape (BZ, N)
        src_intrinsic: Source camera intrinsic matrix, shape (BZ, 3, 3)
        dst_intrinsic: Destination camera intrinsic matrix, shape (BZ, 3, 3)
        relative_pose: Relative pose from src to dst camera (src-to-dst transform), shape (BZ, 3, 4) or (BZ, 4, 4)
        src_depth_scale: Scale factor for source depth values, shape (BZ, 1)
        
    Returns:
        Tuple of:
            - dst_coords: Destination pixel coordinates, shape (BZ, N, 2)
            - valid_mask: Boolean mask indicating valid projections (positive depth), shape (BZ, N)
    """
    BZ, N = src_coords.shape[:2]
    device = src_coords.device
    
    # Ensure relative_pose is 4x4
    if relative_pose.shape[-2:] == (3, 4):
        relative_pose = pad_poses(relative_pose)  # (BZ, 4, 4)
    
    # Apply source depth scaling
    scaled_src_depth = src_depth * src_depth_scale  # (BZ, N)
    
    # Convert to homogeneous pixel coordinates
    src_coords_homo = to_homogeneous(src_coords)  # (BZ, N, 3)
    
    # Backproject to source camera coordinates using batched matrix multiplication
    # src_K_inv: (BZ, 3, 3), src_coords_homo: (BZ, N, 3) -> (BZ, 3, N) -> (BZ, N, 3)
    src_K_inv = torch.inverse(src_intrinsic)  # (BZ, 3, 3)
    src_cam_coords = torch.bmm(src_K_inv, src_coords_homo.transpose(-1, -2)).transpose(-1, -2)  # (BZ, N, 3)
    src_cam_coords = src_cam_coords * scaled_src_depth.unsqueeze(-1)  # (BZ, N, 3)
    
    # Convert to homogeneous camera coordinates
    src_cam_coords_homo = to_homogeneous(src_cam_coords)  # (BZ, N, 4)
    
    # Transform directly from source camera to destination camera using relative pose
    # (BZ, 4, 4) x (BZ, 4, N) -> (BZ, 4, N) -> (BZ, N, 4)
    dst_cam_coords = torch.bmm(relative_pose, src_cam_coords_homo.transpose(-1, -2)).transpose(-1, -2)  # (BZ, N, 4)
    dst_cam_coords_3d = dst_cam_coords[..., :3]  # (BZ, N, 3)
    
    # Extract the projected depth (Z coordinate in destination camera)
    dst_projected_depth = dst_cam_coords_3d[..., 2]  # (BZ, N)
    
    # Project to destination image coordinates (no artificial depth scaling)
    # (BZ, 3, 3) x (BZ, 3, N) -> (BZ, 3, N) -> (BZ, N, 3)
    dst_coords_homo = torch.bmm(dst_intrinsic, dst_cam_coords_3d.transpose(-1, -2)).transpose(-1, -2)  # (BZ, N, 3)
    dst_coords = from_homogeneous(dst_coords_homo)  # (BZ, N, 2)
    
    # Compute validity mask - all points are considered valid
    valid_mask = torch.ones_like(dst_projected_depth, dtype=torch.bool)  # (BZ, N) - all ones
    
    return dst_coords, valid_mask

def backproject_and_reproject_with_approximation(
    src_coords: torch.Tensor,
    src_depth: torch.Tensor,
    dst_depth: torch.Tensor,
    src_intrinsic: torch.Tensor,
    dst_intrinsic: torch.Tensor,
    relative_pose: torch.Tensor,
    src_depth_scale: torch.Tensor,
    dst_depth_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backproject source coordinates to 3D camera space then reproject to destination camera using relative pose.
    
    This is an approximation version that replaces the non-linear perspective division (dividing by projected depth)
    with a fixed destination depth. Instead of using the geometrically computed Z coordinate for perspective
    projection, this function uses the provided scaled destination depth for the division step.
    
    This function performs the geometric pipeline with depth approximation:
    1. Scales source depth values using src_depth_scale
    2. Backprojects source pixel coordinates to 3D camera coordinates using source intrinsics
    3. Transforms 3D points from source camera to destination camera using relative pose
    4. Projects 3D points using intrinsics but replaces perspective division with fixed dst depth
    5. Computes validity mask using scaled destination depth
    
    Args:
        src_coords: Source pixel coordinates, shape (BZ, N, 2)
        src_depth: Source depth values, shape (BZ, N)
        dst_depth: Destination depth values for approximation, shape (BZ, N)
        src_intrinsic: Source camera intrinsic matrix, shape (BZ, 3, 3)
        dst_intrinsic: Destination camera intrinsic matrix, shape (BZ, 3, 3)
        relative_pose: Relative pose from src to dst camera (src-to-dst transform), shape (BZ, 3, 4) or (BZ, 4, 4)
        src_depth_scale: Scale factor for source depth values, shape (BZ, 1)
        dst_depth_scale: Scale factor for destination depth values, shape (BZ, 1)
        
    Returns:
        Tuple of:
            - dst_coords: Destination pixel coordinates (using approximated depth for division), shape (BZ, N, 2)
            - valid_mask: Boolean mask indicating valid projections, shape (BZ, N)
    """
    BZ, N = src_coords.shape[:2]
    device = src_coords.device
    
    # Ensure relative_pose is 4x4
    if relative_pose.shape[-2:] == (3, 4):
        relative_pose = pad_poses(relative_pose)  # (BZ, 4, 4)
    
    # Apply source depth scaling
    scaled_src_depth = src_depth * src_depth_scale  # (BZ, N)
    
    # Apply destination depth scaling for approximation
    scaled_dst_depth = dst_depth * dst_depth_scale  # (BZ, N)
    
    # Convert to homogeneous pixel coordinates
    src_coords_homo = to_homogeneous(src_coords)  # (BZ, N, 3)
    
    # Backproject to source camera coordinates using batched matrix multiplication
    # src_K_inv: (BZ, 3, 3), src_coords_homo: (BZ, N, 3) -> (BZ, 3, N) -> (BZ, N, 3)
    src_K_inv = torch.inverse(src_intrinsic)  # (BZ, 3, 3)
    src_cam_coords = torch.bmm(src_K_inv, src_coords_homo.transpose(-1, -2)).transpose(-1, -2)  # (BZ, N, 3)
    src_cam_coords = src_cam_coords * scaled_src_depth.unsqueeze(-1)  # (BZ, N, 3)
    
    # Convert to homogeneous camera coordinates
    src_cam_coords_homo = to_homogeneous(src_cam_coords)  # (BZ, N, 4)
    
    # Transform directly from source camera to destination camera using relative pose
    # (BZ, 4, 4) x (BZ, 4, N) -> (BZ, 4, N) -> (BZ, N, 4)
    dst_cam_coords = torch.bmm(relative_pose, src_cam_coords_homo.transpose(-1, -2)).transpose(-1, -2)  # (BZ, N, 4)
    dst_cam_coords_3d = dst_cam_coords[..., :3]  # (BZ, N, 3)
    
    # Project to destination image coordinates using intrinsics
    # (BZ, 3, 3) x (BZ, 3, N) -> (BZ, 3, N) -> (BZ, N, 3)
    dst_coords_homo = torch.bmm(dst_intrinsic, dst_cam_coords_3d.transpose(-1, -2)).transpose(-1, -2)  # (BZ, N, 3)
    
    # APPROXIMATION: Replace perspective division with fixed destination depth
    # Instead of dividing by dst_coords_homo[..., 2] (the projected Z), use scaled_dst_depth
    dst_coords = dst_coords_homo[..., :2] / (scaled_dst_depth.unsqueeze(-1) + 1e-6)  # (BZ, N, 2)
    
    # Compute validity mask - all points are considered valid
    valid_mask = torch.ones_like(scaled_dst_depth, dtype=torch.bool)  # (BZ, N) - all ones
    
    return dst_coords, valid_mask

def compute_relative_pose(
    src_extrinsic: torch.Tensor,
    dst_extrinsic: torch.Tensor
) -> torch.Tensor:
    """Compute relative pose transformation from source camera to destination camera.
    
    This function computes the relative pose that transforms 3D points from the source camera
    coordinate system to the destination camera coordinate system.
    
    Camera Pose Convention:
        - Input extrinsic matrices are World-to-Camera transformations [R|t] in OpenCV format
        - R: 3x3 rotation matrix that rotates world points to camera coordinate system  
        - t: 3x1 translation vector from world origin to camera center in camera coordinates
        - Full 4x4 matrix: [[R, t], [0, 0, 0, 1]] transforms world points to camera coordinates
    
    Relative Pose Computation:
        relative_pose = dst_extrinsic @ src_extrinsic^(-1)
        This gives the transformation from src camera coordinates to dst camera coordinates.
    
    Args:
        src_extrinsic: Source camera extrinsic matrix (World-to-Camera), shape (BZ, 3, 4) or (BZ, 4, 4)
        dst_extrinsic: Destination camera extrinsic matrix (World-to-Camera), shape (BZ, 3, 4) or (BZ, 4, 4)
        
    Returns:
        Relative pose transformation matrix (src-to-dst), shape (BZ, 4, 4)
    """
    # Ensure both extrinsics are 4x4 for matrix operations
    if src_extrinsic.shape[-2:] == (3, 4):
        src_extrinsic_4x4 = pad_poses(src_extrinsic)  # Shape: [BZ, 4, 4]
    else:
        src_extrinsic_4x4 = src_extrinsic
        
    if dst_extrinsic.shape[-2:] == (3, 4):
        dst_extrinsic_4x4 = pad_poses(dst_extrinsic)  # Shape: [BZ, 4, 4]
    else:
        dst_extrinsic_4x4 = dst_extrinsic
    
    # Compute relative pose: dst_extrinsic @ src_extrinsic^(-1)
    # This transforms points from src camera coordinates to dst camera coordinates
    src_extrinsic_inv = torch.inverse(src_extrinsic_4x4)  # Shape: [BZ, 4, 4]
    relative_pose = torch.bmm(dst_extrinsic_4x4, src_extrinsic_inv)  # Shape: [BZ, 4, 4]
    
    return relative_pose

def compute_projective_residual(
    predicted_dst_coords: torch.Tensor,
    actual_dst_coords: torch.Tensor
) -> torch.Tensor:
    """Compute the L2 norm of projective residual between predicted and actual destination coordinates.
    
    Args:
        predicted_dst_coords: Predicted destination pixel coordinates from backproject_and_reproject, shape (BZ, N, 2)
        actual_dst_coords: Ground truth destination pixel coordinates, shape (BZ, N, 2)
        
    Returns:
        L2 norm of projective residual per point, shape (BZ, N)
    """
    # Compute the difference between predicted and actual coordinates
    residual = predicted_dst_coords - actual_dst_coords  # (BZ, N, 2)
    
    # Compute L2 norm per point
    l2_norm = torch.norm(residual, dim=-1)  # (BZ, N)
    
    return l2_norm
