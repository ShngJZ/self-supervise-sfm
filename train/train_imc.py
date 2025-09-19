import os
import sys
import math
from datetime import datetime
from pathlib import Path
import socket

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.multiprocessing as mp

# Add parent directory to path for eval imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from sailrecon.models.sail_recon import SailRecon
from datasets.imc2021 import IMC2021, collate_fn
from eval.utils.geometry import save_pointcloud_with_plyfile
from utils.geometry import backproject_and_reproject, backproject_and_reproject_with_approximation, compute_projective_residual, compute_relative_pose
from utils.io import ImagePreprocessor
from utils.sanity_check import sanity_check_relative_poses
from losses.cdf_loss import CDFLossIndexPytorch
from utils.vls import plot_cdf_pdf_curves


def find_free_port(start_port=12362, end_port=13962, max_attempts=100):
    """Find a free port in the given range"""
    for _ in range(max_attempts):
        port = torch.randint(start_port, end_port, (1,)).item()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return str(port)
            except socket.error:
                continue
    raise RuntimeError(f"Could not find a free port after {max_attempts} attempts")


def setup_distributed(rank, world_size, master_port, backend='nccl'):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group with specified backend
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Set the GPU device
    torch.cuda.set_device(rank)
    
    return rank


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with warmup"""
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


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

def prepare_model_input(batch, device):
    """
    Prepare input tensors for the SailRecon model from batch data.
    
    Args:
        batch: Dictionary containing batch data with 'scene_name' and 'rgb_processed'
        device: Target device for tensors
        
    Returns:
        tuple: (duplicated_images, no_reloc_list, reloc_list)
            - duplicated_images: Tensor of shape [2*B, 3, H, W] with anchor and query images
            - no_reloc_list: List of indices for anchor images [0, 1, ..., B-1]
            - reloc_list: List of indices for query images [B, B+1, ..., 2*B-1]
    """
    # Extract scene name for debugging
    scene_name = batch['scene_name']
    
    # Get processed RGB images as stacked tensor
    rgb_processed = batch['rgb_processed']  # Shape: [N, C, H, W]
    
    assert rgb_processed is not None, f"No processed RGB images found in batch for scene: {scene_name}"
    assert rgb_processed.dim() == 4, f"Expected 4D tensor for rgb_processed, got {rgb_processed.dim()}D"
    
    # Move to device
    images = rgb_processed.to(device)
    
    # Duplicate images for anchor/query setup
    duplicated_images = torch.cat([images, images], dim=0)
    original_batch_size = images.shape[0]
    no_reloc_list = list(range(original_batch_size))
    reloc_list = list(range(original_batch_size, 2 * original_batch_size))
    
    return duplicated_images, no_reloc_list, reloc_list

def compute_loss(predictions, batch, device, cdf_loss_module, do_record=False):
    """
    Compute training loss focusing on intrinsic matrix prediction and projective residual using CDF loss.
    
    Args:
        predictions: List of prediction dictionaries from model
        batch: Batch data containing recovery matrices and correspondences
        device: Device for tensor operations
        cdf_loss_module: Pre-initialized CDFLossIndexPytorch module
        do_record: Flag to return additional plotting data
        
    Returns:
        dict: Dictionary containing loss and optionally plotting data
    """
    # Extract predicted intrinsics from predictions (assume batch size 1 and intrinsic always present)
    predicted_intrinsics_batched = torch.cat([pred['intrinsic'] for pred in predictions], dim=0)  # Shape: [N, 3, 3]

    # Move all tensors in batch to device
    batch_on_device = to_device(batch, device)
    
    # Get recovery matrices from batch
    K_prime_to_K = batch_on_device['K_prime_to_K']  # Shape: [N, 3, 3]
    
    # Apply recovery matrix to predicted intrinsics to transform them back to original space
    recovered_intrinsics = torch.bmm(K_prime_to_K, predicted_intrinsics_batched)  # Shape: [N, 3, 3]
    
    # Apply averaging if shared focal is True
    if batch['shared_focal']:
        # Average all recovered intrinsics
        averaged_intrinsic = torch.mean(recovered_intrinsics, dim=0, keepdim=True)  # Shape: [1, 3, 3]
        # Repeat the averaged intrinsic to match the batch size
        recovered_intrinsics = averaged_intrinsic.repeat(recovered_intrinsics.shape[0], 1, 1)  # Shape: [N, 3, 3]
    
    # Extract predicted poses from model predictions
    predicted_poses = torch.cat([pred['extrinsic'] for pred in predictions], dim=0)  # Shape: [N, 3, 4] or [N, 4, 4]
    
    # Get correspondence data (already on device)
    src_coords = batch_on_device['src_coords']  # Shape: [num_pairs, K, 2]
    dst_coords = batch_on_device['dst_coords']  # Shape: [num_pairs, K, 2]
    src_depth = batch_on_device['src_depth']    # Shape: [num_pairs, K]
    dst_depth = batch_on_device['dst_depth']    # Shape: [num_pairs, K]
    src_idx = batch_on_device['src_idx']        # Shape: [num_pairs]
    dst_idx = batch_on_device['dst_idx']        # Shape: [num_pairs]
    
    # Batch operation: get intrinsics and poses for all pairs using advanced indexing
    src_intrinsics = recovered_intrinsics[src_idx]  # Shape: [num_pairs, 3, 3]
    dst_intrinsics = recovered_intrinsics[dst_idx]  # Shape: [num_pairs, 3, 3]
    src_extrinsics = predicted_poses[src_idx]       # Shape: [num_pairs, 3, 4] or [num_pairs, 4, 4]
    dst_extrinsics = predicted_poses[dst_idx]       # Shape: [num_pairs, 3, 4] or [num_pairs, 4, 4]
    
    # Compute relative pose from src to dst camera using the dedicated function
    relative_poses = compute_relative_pose(src_extrinsics, dst_extrinsics)  # Shape: [num_pairs, 4, 4]

    # Use unit depth scales (no scaling applied) - shape matches batch size
    num_pairs = src_coords.shape[0]
    src_depth_scale = torch.ones(num_pairs, 1, device=device, dtype=torch.float32)  # Shape: [num_pairs, 1]
    dst_depth_scale = torch.ones(num_pairs, 1, device=device, dtype=torch.float32)  # Shape: [num_pairs, 1]
    
    # Backproject and reproject to get predicted destination coordinates (batch operation)
    predicted_dst_coords, valid_mask = backproject_and_reproject(
        src_coords=src_coords,
        src_depth=src_depth,
        src_intrinsic=src_intrinsics,
        dst_intrinsic=dst_intrinsics,
        relative_pose=relative_poses,
        src_depth_scale=src_depth_scale
    )  # Shape: [num_pairs, K, 2], [num_pairs, K]
    
    # Compute projective residuals (batch operation)
    residuals = compute_projective_residual(predicted_dst_coords, dst_coords)  # Shape: [num_pairs, K]
    
    # Apply validity mask to filter out invalid projections
    residuals = residuals * valid_mask.float()  # Zero out invalid projections
    
    # APPROXIMATION-BASED PROJECTIVE RESIDUAL COMPUTATION
    # Backproject and reproject using approximation with actual dst_depth (batch operation)
    predicted_dst_coords_approx, valid_mask_approx = backproject_and_reproject_with_approximation(
        src_coords=src_coords,
        src_depth=src_depth,
        dst_depth=dst_depth,
        src_intrinsic=src_intrinsics,
        dst_intrinsic=dst_intrinsics,
        relative_pose=relative_poses,
        src_depth_scale=src_depth_scale,
        dst_depth_scale=dst_depth_scale
    )  # Shape: [num_pairs, K, 2], [num_pairs, K]
    
    # Compute approximation-based projective residuals (batch operation)
    residuals_approx = compute_projective_residual(predicted_dst_coords_approx, dst_coords)  # Shape: [num_pairs, K]
    
    # Apply validity mask to filter out invalid projections
    residuals_approx = residuals_approx * valid_mask_approx.float()  # Zero out invalid projections
    
    # Transform residuals with log(1 + x) to handle large values
    residuals_log = torch.log1p(residuals)  # log(1 + x)
    residuals_approx_log = torch.log1p(residuals_approx)  # log(1 + x)
    
    # Create weights (use validity masks)
    weights = valid_mask.float()
    weights_approx = valid_mask_approx.float()
    
    # Compute CDF loss for regular residuals
    cdf_out_src, cdf_out_dst = cdf_loss_module(residuals_log, weights)
    cdf_loss_regular = (cdf_out_src.mean() + cdf_out_dst.mean()) / 2.0
    
    # Compute CDF loss for approximation residuals  
    cdf_out_src_approx, cdf_out_dst_approx = cdf_loss_module(residuals_approx_log, weights_approx)
    cdf_loss_approx = (cdf_out_src_approx.mean() + cdf_out_dst_approx.mean()) / 2.0
    
    # Final loss is the mean of both CDF losses
    total_loss = (cdf_loss_regular + cdf_loss_approx) / 2.0
    
    # Always return loss in dictionary
    result = {"loss": total_loss}
    
    # Optionally add plotting data
    if do_record:
        # Get frame statistics for plotting (use non-approximation residuals only)
        stats = cdf_loss_module.get_frame_statistics(residuals_log.detach(), weights)
        result["plot_data"] = {
            "frame_cdfs": stats['frame_cdf'],
            "frame_pdfs": stats['frame_pdf'],
            "max_val": cdf_loss_module.max_val,
            "min_val": cdf_loss_module.min_val,
            "num_bins": cdf_loss_module.num_bins
        }
    
    return result



def save_checkpoint(model, optimizer, scheduler, scaler, step, loss, checkpoint_dir, is_distributed=False):
    """Save model checkpoint (model weights only)"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_state_dict = model.module.state_dict() if is_distributed else model.state_dict()
    
    # Save only model state dict for inference use
    checkpoint_path = checkpoint_dir / f"model_step_{step}.pt"
    torch.save(model_state_dict, checkpoint_path)
    
    latest_path = checkpoint_dir / "model_latest.pt"
    torch.save(model_state_dict, latest_path)
    
    return checkpoint_path


def save_visualization(predictions, step, results_dir):
    """
    Save point cloud visualization for the given predictions and training step.
    
    Args:
        predictions: Model predictions containing point cloud data
        step: Current training step
        results_dir: Base results directory for saving outputs
        
    Returns:
        str: Path to the saved PLY file
    """
    from eval.utils.device import to_cpu
    
    # Create visualization directory
    vls_dir = os.path.join(results_dir, "vls")
    os.makedirs(vls_dir, exist_ok=True)
    
    # Save the predicted point cloud and camera poses
    scene_output_dir = os.path.join(vls_dir, f"step_{step}")
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # Convert predictions to CPU (same as demo scripts)
    predictions_cpu = [to_cpu(pred) for pred in predictions]
    
    ply_path = os.path.join(scene_output_dir, "pred.ply")
    save_pointcloud_with_plyfile(predictions_cpu, ply_path)
    
    return ply_path


def train_epoch(model, dataloader, optimizer, scheduler, scaler, params, rank, start_step=0):
    """Train for one epoch"""
    model.train()
    
    device = params['device']
    dtype = params['dtype']
    world_size = params['world_size']
    writer = params.get('writer')
    results_dir = params['results_dir']
    
    # Create CDF loss module with specified hyperparameters
    # Use dummy indices for now - they will be constructed properly inside the CDF loss module
    max_val = 15.0
    num_bins = 250
    gradient_smooth = 0.05
    min_val = 0.0
    
    # Create dummy indices for initialization (actual indices are constructed inside the module)
    src_indices = torch.tensor([0], device=device)
    dst_indices = torch.tensor([0], device=device)
    
    cdf_loss_module = CDFLossIndexPytorch(
        min_val=min_val,
        max_val=max_val, 
        num_bins=num_bins,
        src_indices=src_indices,
        dst_indices=dst_indices,
        gradient_smooth=gradient_smooth
    ).to(device)
    
    step = start_step
    
    pbar = tqdm(dataloader, desc=f"Training (Rank {rank})", disable=(rank != 0))
    
    for batch_idx, batch in enumerate(pbar):
        step += 1
        
        # Prepare model input using the dedicated function
        duplicated_images, no_reloc_list, reloc_list = prepare_model_input(batch, device)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()

        with autocast(dtype=dtype):
            predictions = model.forward(
                duplicated_images, 
                no_reloc_list=no_reloc_list, 
                reloc_list=reloc_list
            )
        
        # # Perform sanity check on relative poses before computing loss
        # sanity_check_relative_poses(predictions, batch, device, save_dir=os.path.join(results_dir, "sanity_check"))
        
        # Compute loss outside autocast for full precision numerical stability
        # Enable recording every 10k steps when saving checkpoints
        do_record = (step % 10000 == 1 and rank == 0)
        loss_result = compute_loss(predictions, batch, device, cdf_loss_module, do_record)
        loss = loss_result["loss"]

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        current_lr = scheduler.step()

        # Update progress bar and log
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{current_lr:.2e}',
                'step': step
            })

            # Log to tensorboard
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/learning_rate', current_lr, step)

        # Save checkpoint every 10k steps
        if do_record:
            # Create checkpoint directory under results_dir
            checkpoint_dir = os.path.join(results_dir, f"checkpoint_step_{step}")
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, scaler, step, loss.item(),
                checkpoint_dir, is_distributed=(world_size > 1)
            )
            print(f"Saved checkpoint at step {step}: {checkpoint_path}")
            
            # Save visualization using the dedicated function
            ply_path = save_visualization(predictions, step, results_dir)
            print(f"Saved point cloud visualization at step {step}: {ply_path}")
            
            # Generate CDF/PDF plots
            plot_data = loss_result["plot_data"]
            plot_save_path = os.path.join(results_dir, "plots", f"cdf_pdf_step_{step}.png")
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
            
            plot_cdf_pdf_curves(
                frame_cdfs=plot_data["frame_cdfs"],
                frame_pdfs=plot_data["frame_pdfs"],
                min_val=plot_data["min_val"],
                max_val=plot_data["max_val"],
                num_bins=plot_data["num_bins"],
                save_path=plot_save_path
            )
            print(f"Saved CDF/PDF plots at step {step}: {plot_save_path}")
        a = 1
        break
    return step


def train_worker(rank, world_size, master_port, params):
    """Worker function for distributed training"""
    # Setup distributed training for this process
    backend = params.get('backend', 'nccl')
    setup_distributed(rank, world_size, master_port, backend)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    
    # Set dtype based on GPU capability (same as demo scripts)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    if rank == 0:
        print(f"Using mixed precision dtype: {dtype}")
    
    # Create results directory with timestamp (only on rank 0)
    results_dir = params['results_dir']
    
    # Initialize tensorboard writer (only on rank 0)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(results_dir, 'tensorboard'))
    
    # Add runtime parameters to params dict
    params.update({
        'device': device,
        'dtype': dtype,
        'world_size': world_size,
        'writer': writer
    })
    
    # Initialize model and load pretrained weights for testing
    _URL = "https://huggingface.co/HKUST-SAIL/SAIL-Recon/resolve/main/sailrecon.pt"
    model = SailRecon(kv_cache=False)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(_URL)
    )
    model = model.to(device)
    
    # Always wrap model with DDP since we assume world_size > 1
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['max_lr'], betas=(0.9, 0.999), eps=1e-8)
    
    # Initialize learning rate scheduler
    total_steps = params['epochs'] * 1000  # Estimate total steps
    scheduler = CosineWarmupScheduler(
        optimizer, 
        warmup_steps=params['warmup_steps'],
        max_steps=total_steps,
        max_lr=params['max_lr'],
        min_lr=params['max_lr'] * 0.01
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Initialize dataset and dataloader
    dataset = IMC2021(root=params['data_root'], num_images=params['num_images'])
    
    # Always use distributed sampler since we assume world_size > 1
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # Always False when using DistributedSampler
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Dataset contains {len(dataset)} scenes")
        print(f"Training on {world_size} GPU(s)")
        print(f"Results directory: {results_dir}")
    
    # Training loop
    current_step = 0
    
    for epoch in range(params['epochs']):
        sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{params['epochs']}")
        
        # Train for one epoch
        current_step = train_epoch(
            model, dataloader, optimizer, scheduler, scaler, 
            params, rank, start_step=current_step
        )
        
        if rank == 0:
            print(f"Epoch {epoch + 1} completed.")
    
    # Save final checkpoint
    if rank == 0:
        final_checkpoint = save_checkpoint(
            model, optimizer, scheduler, scaler, current_step, 0.0,
            results_dir, is_distributed=True
        )
        print(f"Saved final checkpoint: {final_checkpoint}")
        
        if writer:
            writer.close()
    
    # Clean up distributed training
    dist.destroy_process_group()


def main():
    # Configuration with backend options
    params = {
        'data_root': "/home/ubuntu/disk6/Motion-from-Structure/release/imc2021/DUSt3R_RoMa",
        'max_lr': 2e-4,
        'warmup_steps': 2000,
        'epochs': 100,
        'backend': 'nccl',  # Use NCCL for better performance with CUDA
        'num_images': 2,  # Number of images to select per scene
    }
    
    # Get world_size by counting available GPUs
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"/home/ubuntu/disk6/self-supervise-sfm/results/imc_{timestamp}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    params['results_dir'] = results_dir
    
    # Find free port for distributed training
    master_port = find_free_port()
    
    backend = params.get('backend', 'nccl')
    print(f"Starting distributed training on {world_size} GPUs with backend '{backend}' and master port {master_port}")
    
    # Use mp.spawn for distributed training
    mp.spawn(
        train_worker,
        args=(world_size, master_port, params),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()


