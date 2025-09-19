import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class CDFLossTorchWrapper(torch.autograd.Function):
    """PyTorch autograd wrapper for CDF loss."""
    @staticmethod
    def forward(ctx, residuals, residuals_cdf, residuals_cdf_grad):
        ctx.save_for_backward(residuals_cdf_grad)
        return residuals_cdf

    @staticmethod
    def backward(ctx, grad_out):
        residuals_cdf_grad, = ctx.saved_tensors
        return residuals_cdf_grad * grad_out, None, None


class CDFLossIndexPytorch(torch.nn.Module):
    """
    Fully vectorized PyTorch implementation of CDFLossIndexCupy.
    
    No for loops - uses scatter_add and gather for maximum efficiency.
    Each frame gets its own CDF built from all pairs involving that frame.
    """
    
    def __init__(
        self,
        min_val: float,
        max_val: float,
        num_bins: int,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        gradient_smooth: float = 0.0001,  # Physical smoothness radius like reference
        num_nodes: Optional[int] = None,
    ):
        super().__init__()
        
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
        self.gradient_smooth = gradient_smooth
        
        # Register buffers
        self.register_buffer('src_indices', src_indices.long())
        self.register_buffer('dst_indices', dst_indices.long())
        
        # Compute number of nodes
        if num_nodes is None:
            all_indices = torch.cat([src_indices, dst_indices])
            self.num_nodes = int(torch.max(all_indices).item()) + 1
        else:
            self.num_nodes = num_nodes
            
        # Two-step approach: gradient computation + optional smoothing
        bin_width = (max_val - min_val) / num_bins
        
        # Step 1: Always use simple Sobel filter for gradient computation (CDF → raw PDF)
        self.grad_conv = torch.nn.Conv1d(1, 1, 3, padding=1, bias=False, padding_mode="reflect")
        sobel_kernel = torch.tensor([-1.0, 0.0, 1.0]) / (2.0 * bin_width)
        self.grad_conv.weight.data[0, 0, :] = sobel_kernel
        self.grad_conv.weight.requires_grad = False
        
        # Step 2: Smoothing convolution (identity by default, Gaussian if smoothing requested)
        if gradient_smooth > 0:
            # Gaussian smoothing kernel
            kernel_radius_bins = max(1, int(gradient_smooth / bin_width))
            kernel_size = 2 * kernel_radius_bins + 1
            
            self.smooth_conv = torch.nn.Conv1d(
                1, 1, kernel_size, 
                padding=kernel_radius_bins, bias=False, padding_mode="reflect"
            )
            
            # Standard Gaussian kernel for smoothing (bell-shaped, peaked at center)
            kernel_indices = torch.arange(kernel_size, dtype=torch.float32) - kernel_radius_bins
            sigma = gradient_smooth / bin_width
            
            gaussian = torch.exp(-0.5 * (kernel_indices / sigma) ** 2)
            gaussian_kernel = gaussian / torch.sum(gaussian)  # Normalize to sum = 1
            
            self.smooth_conv.weight.data[0, 0, :] = gaussian_kernel
        else:
            # Identity smoothing (no smoothing)
            self.smooth_conv = torch.nn.Conv1d(1, 1, 1, padding=0, bias=False)
            self.smooth_conv.weight.data[0, 0, 0] = 1.0  # Identity kernel
        
        self.smooth_conv.weight.requires_grad = False

    def forward(
        self, 
        residuals: torch.Tensor, 
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CDF values using fully vectorized operations.
        
        Args:
            residuals: [N_pairs, K_points] 
            weights: [N_pairs, K_points]
            
        Returns:
            Tuple of (src_cdf, dst_cdf) with same shape as residuals
        """
        # Step 1: Build per-frame histograms (vectorized)
        pmf, cdf, pdf = self.compute_weighted_pdf_cdf_vectorized(residuals, weights)
        
        # Step 2: Lookup CDF values (vectorized)
        residuals_cdf_src, residuals_cdf_grad_src, residuals_cdf_dst, residuals_cdf_grad_dst = \
            self.compute_weighted_cdf_forward_backward_vectorized(residuals, weights, cdf, pdf)
        
        # Step 3: Apply autograd wrapper
        residuals_cdf_out_src = CDFLossTorchWrapper.apply(
            residuals, residuals_cdf_src, residuals_cdf_grad_src
        )
        residuals_cdf_out_dst = CDFLossTorchWrapper.apply(
            residuals, residuals_cdf_dst, residuals_cdf_grad_dst
        )
        
        return residuals_cdf_out_src, residuals_cdf_out_dst

    @torch.no_grad()
    def compute_weighted_pdf_cdf_vectorized(
        self, 
        residuals: torch.Tensor, 
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized histogram building - no for loops!
        
        Key insight: Each residual contributes to TWO frame histograms.
        We flatten everything and use scatter_add efficiently.
        """
        device = residuals.device
        num_pairs, num_points = residuals.shape
        
        # Compute bin indices for all residuals [N_pairs, K_points]
        bin_width = (self.max_val - self.min_val) / self.num_bins
        bin_indices = ((residuals - self.min_val) / bin_width).long()
        valid_mask = (bin_indices >= 0) & (bin_indices < self.num_bins)
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        # Apply weights and mask [N_pairs, K_points]
        weighted_contributions = weights * valid_mask.float()
        
        # **VECTORIZED HISTOGRAM BUILDING**
        # Create flattened indices for scatter_add
        
        # Flatten residuals and weights
        flat_bin_indices = bin_indices.flatten()  # [N_pairs * K_points]
        flat_weights = weighted_contributions.flatten()  # [N_pairs * K_points]
        
        # Create pair indices for each residual [N_pairs * K_points]
        pair_indices = torch.arange(num_pairs, device=device).unsqueeze(1).expand(-1, num_points).flatten()
        
        # Get frame indices for each residual
        src_frame_indices = self.src_indices[pair_indices]  # [N_pairs * K_points]
        dst_frame_indices = self.dst_indices[pair_indices]  # [N_pairs * K_points]
        
        # Create combined indices for scatter_add: frame_idx * num_bins + bin_idx
        src_combined_idx = src_frame_indices * self.num_bins + flat_bin_indices  # [N_pairs * K_points]
        dst_combined_idx = dst_frame_indices * self.num_bins + flat_bin_indices  # [N_pairs * K_points]
        
        # Initialize flat histogram and use scatter_add (VECTORIZED!)
        flat_hist = torch.zeros(self.num_nodes * self.num_bins, device=device, dtype=torch.float32)
        
        # Add contributions from source and destination perspectives
        flat_hist.scatter_add_(0, src_combined_idx, flat_weights)
        flat_hist.scatter_add_(0, dst_combined_idx, flat_weights)
        
        # Reshape back to [num_nodes, num_bins]
        frame_hist = flat_hist.view(self.num_nodes, self.num_bins)
        
        # Compute total weight per frame (including out-of-range residuals for proper normalization)
        total_weight_per_frame = torch.zeros(self.num_nodes, device=device, dtype=torch.float32)
        
        # Add ALL weights (both valid and invalid residuals) to get proper totals
        total_weight_per_frame.scatter_add_(0, src_frame_indices, weights.flatten())
        total_weight_per_frame.scatter_add_(0, dst_frame_indices, weights.flatten())
        
        # Compute PMF with correct normalization (total weights, not just histogram sum)
        pmf = frame_hist / (total_weight_per_frame.unsqueeze(1) + 1e-10)  # Probability Mass Function (normalized histogram)
        cdf = torch.cumsum(pmf, dim=1)  # Cumulative Distribution Function
        
        # Two-step PDF computation: gradient + smoothing (identity if no smoothing requested)
        raw_pdf = self.grad_conv(cdf.view(self.num_nodes, 1, self.num_bins)).view(self.num_nodes, self.num_bins)
        pdf = self.smooth_conv(raw_pdf.view(self.num_nodes, 1, self.num_bins)).view(self.num_nodes, self.num_bins)
        
        return pmf, cdf, pdf

    @torch.no_grad()
    def compute_weighted_cdf_forward_backward_vectorized(
        self,
        residuals: torch.Tensor,
        weights: torch.Tensor,
        cdf: torch.Tensor,
        pdf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized CDF lookup - no for loops!
        
        Uses gather operations to lookup all CDF values at once.
        """
        num_pairs, num_points = residuals.shape
        device = residuals.device
        
        # Compute bin indices with +0.5 rounding (matches CuPy)
        bin_width = (self.max_val - self.min_val) / self.num_bins
        bin_indices = ((residuals - self.min_val) / bin_width + 0.5).long()
        valid_mask = (bin_indices >= 0) & (bin_indices < self.num_bins) & (weights > 0.0)
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        # **VECTORIZED CDF LOOKUP**
        
        # Expand frame indices to match residuals shape [N_pairs, K_points]
        src_frame_expanded = self.src_indices.unsqueeze(1).expand(-1, num_points)  # [N_pairs, K_points]
        dst_frame_expanded = self.dst_indices.unsqueeze(1).expand(-1, num_points)  # [N_pairs, K_points]
        
        # Create indices for gather: [frame_idx, bin_idx] -> frame_idx * num_bins + bin_idx
        src_gather_idx = src_frame_expanded * self.num_bins + bin_indices  # [N_pairs, K_points]
        dst_gather_idx = dst_frame_expanded * self.num_bins + bin_indices  # [N_pairs, K_points]
        
        # Flatten CDF and PDF for gather
        flat_cdf = cdf.flatten()  # [num_nodes * num_bins]
        flat_pdf = pdf.flatten()  # [num_nodes * num_bins]
        
        # Gather CDF values (VECTORIZED!)
        src_cdf_values = flat_cdf.gather(0, src_gather_idx.flatten()).view(num_pairs, num_points)
        dst_cdf_values = flat_cdf.gather(0, dst_gather_idx.flatten()).view(num_pairs, num_points)
        
        # Gather PDF values and apply weights (VECTORIZED!)
        src_grad_values = flat_pdf.gather(0, src_gather_idx.flatten()).view(num_pairs, num_points) * weights
        dst_grad_values = flat_pdf.gather(0, dst_gather_idx.flatten()).view(num_pairs, num_points) * weights
        
        # Apply masks with default values (matches CuPy: 2.0 for invalid CDF, 0.0 for invalid grad)
        residuals_cdf_src = torch.where(valid_mask, src_cdf_values, torch.full_like(src_cdf_values, 2.0))
        residuals_cdf_dst = torch.where(valid_mask, dst_cdf_values, torch.full_like(dst_cdf_values, 2.0))
        
        residuals_cdf_grad_src = torch.where(valid_mask, src_grad_values, torch.zeros_like(src_grad_values))
        residuals_cdf_grad_dst = torch.where(valid_mask, dst_grad_values, torch.zeros_like(dst_grad_values))
        
        return residuals_cdf_src, residuals_cdf_grad_src, residuals_cdf_dst, residuals_cdf_grad_dst

    @torch.no_grad()
    def get_frame_statistics(
        self, 
        residuals: torch.Tensor, 
        weights: torch.Tensor
    ) -> dict:
        """Get per-frame histogram statistics for debugging."""
        pmf, cdf, pdf = self.compute_weighted_pdf_cdf_vectorized(residuals, weights)
        
        return {
            'frame_pmf': pmf.cpu().numpy(),  # Normalized histogram (PMF)
            'frame_cdf': cdf.cpu().numpy(),
            'frame_pdf': pdf.cpu().numpy(),  # grad_cdf is the actual PDF
            'num_nodes': self.num_nodes,
            'pairs': list(zip(self.src_indices.cpu().numpy(), self.dst_indices.cpu().numpy()))
        }


# Simple uniform distribution test
if __name__ == "__main__":
    def create_ground_truth_distributions(K: int, device: torch.device, max_val: float = 0.01):
        """Create K different known non-negative distributions for testing."""
        distributions = []

        for k in range(K):
            # Create different types of distributions
            if k % 4 == 0:
                # Exponential distribution: f(x) = λe^(-λx)
                rate = 20.0 + k * 10  # Different rates
                dist_type = "exponential"
                params = {"rate": rate}
            elif k % 4 == 1:
                # Gamma distribution: more peaked
                shape, rate = 2.0 + k * 0.5, 100.0 + k * 20
                dist_type = "gamma"
                params = {"shape": shape, "rate": rate}
            elif k % 4 == 2:
                # Uniform distribution in different ranges
                low, high = k * max_val / K, (k + 1) * max_val / K
                dist_type = "uniform"
                params = {"low": low, "high": high}
            else:
                # Beta distribution (scaled to [0, max_val])
                alpha, beta = 0.5 + k * 0.3, 2.0 + k * 0.5
                dist_type = "beta"
                params = {"alpha": alpha, "beta": beta}

            distributions.append({
                "type": dist_type,
                "params": params,
                "id": k
            })

        return distributions


    def sample_from_distribution(dist_info: dict, num_samples: int, device: torch.device, max_val: float = 0.01):
        """Sample from a specific distribution."""
        dist_type = dist_info["type"]
        params = dist_info["params"]

        if dist_type == "exponential":
            # Exponential: use inverse transform sampling
            u = torch.rand(num_samples, device=device)
            samples = -torch.log(1 - u) / params["rate"]
        elif dist_type == "gamma":
            # Gamma distribution
            samples = torch.distributions.Gamma(params["shape"], params["rate"]).sample((num_samples,)).to(device)
        elif dist_type == "uniform":
            # Uniform distribution
            samples = torch.rand(num_samples, device=device) * (params["high"] - params["low"]) + params["low"]
        elif dist_type == "beta":
            # Beta distribution scaled to [0, max_val]
            beta_samples = torch.distributions.Beta(params["alpha"], params["beta"]).sample((num_samples,)).to(device)
            samples = beta_samples * max_val
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        # Allow out-of-range values for testing gradient behavior
        return samples


    def compute_theoretical_cdf(x: torch.Tensor, dist_info: dict, max_val: float = 0.01):
        """Compute theoretical CDF for comparison."""
        dist_type = dist_info["type"]
        params = dist_info["params"]

        if dist_type == "exponential":
            # CDF: F(x) = 1 - e^(-λx)
            cdf_vals = 1 - torch.exp(-params["rate"] * x)
        elif dist_type == "gamma":
            # Use scipy for gamma CDF (approximate)
            import scipy.stats as stats
            x_np = x.cpu().numpy()
            cdf_np = stats.gamma.cdf(x_np, a=params["shape"], scale=1 / params["rate"])
            cdf_vals = torch.tensor(cdf_np, device=x.device, dtype=x.dtype)
        elif dist_type == "uniform":
            # Uniform CDF: F(x) = (x - a) / (b - a)
            cdf_vals = torch.clamp((x - params["low"]) / (params["high"] - params["low"]), 0, 1)
        elif dist_type == "beta":
            # Beta CDF scaled to [0, max_val]
            import scipy.stats as stats
            x_scaled = (x / max_val).cpu().numpy()
            cdf_np = stats.beta.cdf(x_scaled, params["alpha"], params["beta"])
            cdf_vals = torch.tensor(cdf_np, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        return torch.clamp(cdf_vals, 0, 1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Simple parameters with narrow range to force out-of-border cases
    num_frames = 4
    num_pairs = 12
    points_per_pair = 50000
    max_val = 0.2  # Reduced from 1.0 to create many out-of-border cases
    num_bins = 5000
    
    print(f"Testing complex distributions: {num_frames} frames, {num_pairs} pairs, {points_per_pair} points")
    
    # Create complex non-negative distributions
    distributions = [
        {"type": "exponential", "params": {"rate": 50.0}},  # Fast decay
        {"type": "gamma", "params": {"shape": 2.0, "rate": 100.0}},  # Peaked distribution
        {"type": "exponential", "params": {"rate": 20.0}},  # Slower decay  
        {"type": "gamma", "params": {"shape": 3.0, "rate": 150.0}}   # More peaked
    ]
    
    # Create pairs (each frame maps to itself)
    src_indices = torch.tensor([i // (num_pairs // num_frames) for i in range(num_pairs)], device=device)
    dst_indices = src_indices.clone()
    
    # Sample from complex distributions
    residuals = torch.zeros(num_pairs, points_per_pair, device=device)
    for i in range(num_pairs):
        frame_id = src_indices[i].item()
        dist_info = distributions[frame_id]
        samples = sample_from_distribution(dist_info, points_per_pair, device, max_val)
        residuals[i] = samples
    
    # Run CDF loss with absolute smoothing range (independent of histogram resolution)
    weights = torch.ones_like(residuals)
    residuals.requires_grad_(True)
    gradient_smooth = 0.003  # Absolute x-axis smoothing range (independent of num_bins)
    cdf_loss = CDFLossIndexPytorch(0.0, max_val, num_bins, src_indices, dst_indices,
                                   gradient_smooth=gradient_smooth, num_nodes=num_frames).to(device)
    
    # Forward pass
    cdf_out_src, cdf_out_dst = cdf_loss(residuals, weights)
    
    # Backward pass to test gradients
    # Note: divide by 2 since src and dst are identical in this test case
    loss = (cdf_out_src.sum() + cdf_out_dst.sum()) / 2.0
    loss.backward()
    
    # Get frame statistics for plotting
    stats = cdf_loss.get_frame_statistics(residuals.detach(), weights)
    frame_pdfs = stats['frame_pdf']
    frame_cdfs = stats['frame_cdf']
    
    # Plot results
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    output_dir = "/home/ubuntu/tmp"
    os.makedirs(output_dir, exist_ok=True)
    
    bin_centers = np.linspace(0, max_val, num_bins)
    
    fig, axes = plt.subplots(2, num_frames, figsize=(16, 8))
    fig.suptitle('CDF Loss: Complex Distribution Recovery', fontsize=16)
    
    for frame_id in range(num_frames):
        dist_info = distributions[frame_id]
        dist_type = dist_info["type"]
        params = dist_info["params"]
        
        # Compute theoretical CDF and PDF for complex distributions
        x_tensor = torch.tensor(bin_centers, device=device, dtype=torch.float32)
        theoretical_cdf_tensor = compute_theoretical_cdf(x_tensor, dist_info, max_val)
        theoretical_cdf = theoretical_cdf_tensor.cpu().numpy()
        
        # Compute theoretical PDF by differentiating CDF
        if dist_type == "exponential":
            # PDF: f(x) = λe^(-λx)
            theoretical_pdf = params["rate"] * np.exp(-params["rate"] * bin_centers)
        elif dist_type == "gamma":
            # Use scipy for gamma PDF
            import scipy.stats as stats
            theoretical_pdf = stats.gamma.pdf(bin_centers, a=params["shape"], scale=1/params["rate"])
        else:
            # Fallback: numerical derivative of CDF
            theoretical_pdf = np.gradient(theoretical_cdf, bin_centers[1] - bin_centers[0])
        
        # Recovered distributions
        recovered_pdf = frame_pdfs[frame_id]
        recovered_cdf = frame_cdfs[frame_id]
        
        # Create descriptive title
        if dist_type == "exponential":
            title_str = f'Exp(λ={params["rate"]:.1f})'
        elif dist_type == "gamma":
            title_str = f'Gamma(α={params["shape"]:.1f}, β={params["rate"]:.1f})'
        else:
            title_str = f'{dist_type.title()}'
        
        # CDF plot
        axes[0, frame_id].plot(bin_centers, theoretical_cdf, 'b-', linewidth=2, label='Theoretical')
        axes[0, frame_id].plot(bin_centers, recovered_cdf, 'r--', linewidth=2, label='Recovered')
        axes[0, frame_id].set_title(f'Frame {frame_id}: CDF {title_str}')
        axes[0, frame_id].legend()
        axes[0, frame_id].grid(True, alpha=0.3)
        axes[0, frame_id].set_xlim(0, min(0.1, max_val))  # Focus on relevant range
        
        # PDF plot
        axes[1, frame_id].plot(bin_centers, theoretical_pdf, 'b-', linewidth=2, label='Theoretical')
        axes[1, frame_id].plot(bin_centers, recovered_pdf, 'r--', linewidth=2, label='Recovered')
        axes[1, frame_id].set_title(f'Frame {frame_id}: PDF {title_str}')
        axes[1, frame_id].legend()
        axes[1, frame_id].grid(True, alpha=0.3)
        axes[1, frame_id].set_xlim(0, min(0.1, max_val))  # Focus on relevant range
        
        # Calculate errors
        cdf_mae = np.mean(np.abs(recovered_cdf - theoretical_cdf))
        pdf_mae = np.mean(np.abs(recovered_pdf - theoretical_pdf))
        print(f"Frame {frame_id} ({title_str}): CDF MAE = {cdf_mae:.6f}, PDF MAE = {pdf_mae:.6f}")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/complex_recovery.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_dir}/complex_recovery.png")
    
    # Test gradient vs theoretical PDF comparison
    print("\n=== Gradient vs Theoretical PDF Comparison ===")
    torch.manual_seed(123)  # For reproducible sampling
    sample_indices = torch.randint(0, num_pairs, (3,))  # Sample 3 random pairs
    sample_points = torch.randint(0, points_per_pair, (5,))  # Sample 5 random points per pair

    gradient_pdf_errors = []
    for i, pair_idx in enumerate(sample_indices):
        frame_id = src_indices[pair_idx].item()
        dist_info = distributions[frame_id]
        
        for j, point_idx in enumerate(sample_points):
            residual_val = residuals[pair_idx, point_idx].detach().item()
            gradient_val = residuals.grad[pair_idx, point_idx].item()
            
            # Compute theoretical PDF at this point
            if dist_info["type"] == "exponential":
                theoretical_pdf_val = dist_info["params"]["rate"] * np.exp(-dist_info["params"]["rate"] * residual_val)
            elif dist_info["type"] == "gamma":
                import scipy.stats as stats
                theoretical_pdf_val = stats.gamma.pdf(residual_val, a=dist_info["params"]["shape"], scale=1/dist_info["params"]["rate"])
            else:
                continue  # Skip other distributions
            
            # Compare gradient with theoretical PDF (d/dx CDF(x) = PDF(x))
            if 0.0 <= residual_val <= max_val and theoretical_pdf_val > 1e-6:
                error = abs(gradient_val - theoretical_pdf_val)
                relative_error = error / max(theoretical_pdf_val, 1e-6)
                gradient_pdf_errors.append(relative_error)

                print(f"Sample {len(gradient_pdf_errors)}: x={residual_val:.4f}, "
                      f"grad={gradient_val:.6f}, theoretical_pdf={theoretical_pdf_val:.6f}, "
                      f"rel_error={relative_error:.4f}")
            
    # Simple out-of-boundary gradient check
    print("\n=== Out-of-Boundary Gradient Check ===")
    out_of_bounds = (residuals.detach() < 0.0) | (residuals.detach() > max_val)
    out_of_bounds_count = torch.sum(out_of_bounds).item()
    
    if out_of_bounds_count > 0:
        out_gradients = residuals.grad[out_of_bounds]
        zero_grads = torch.sum(torch.abs(out_gradients) < 1e-6).item()
        print(f"Out-of-bounds residuals: {out_of_bounds_count}")
        print(f"With zero gradient: {zero_grads} ({100.0*zero_grads/out_of_bounds_count:.1f}%)")
        print(f"Max gradient magnitude: {torch.max(torch.abs(out_gradients)).item():.8f}")


