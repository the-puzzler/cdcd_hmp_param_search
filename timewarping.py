import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeWarping(nn.Module):
    def __init__(self, num_bins, t_min=0, t_max=1, ema_decay=0.99):
        super().__init__()
        self.num_bins = num_bins
        self.t_min = t_min
        self.t_max = t_max
        self.ema_decay = ema_decay
        self.eps = 1e-6
        
        # Initialize logits for input and output bins to -log(N)
        init_value = -torch.log(torch.tensor(num_bins))
        self.register_parameter('input_logits', nn.Parameter(init_value.repeat(num_bins)))
        self.register_parameter('output_logits', nn.Parameter(init_value.repeat(num_bins)))
        
        # Register EMA buffers
        self.register_buffer('input_logits_ema', init_value.repeat(num_bins))
        self.register_buffer('output_logits_ema', init_value.repeat(num_bins))

        # Add buffers for collecting epoch statistics
        self.register_buffer('epoch_loss_sum', torch.zeros(num_bins))
        self.register_buffer('epoch_count', torch.zeros(num_bins))

    def get_bin_edges(self, logits, normalize=True):
        """Compute bin edges and weights from logits"""
        if normalize:
            weights = F.softmax(logits, dim=0)
        else:
            weights = torch.exp(logits)
            
        # Add small constant and renormalize for stability
        weights = weights + self.eps
        if normalize:
            weights = weights / weights.sum()
            
        # Compute cumulative sum for bin edges
        edges = torch.cat([
            torch.zeros(1, device=logits.device),
            torch.cumsum(weights, dim=0)
        ])
        
        return edges, weights
    
    def normalize_time(self, t):
        """Normalize time to [0, 1] interval (Equation 10)"""
        return (t - self.t_min) / (self.t_max - self.t_min)
    
    def denormalize_time(self, t_norm):
        """Convert normalized time back to original scale"""
        return t_norm * (self.t_max - self.t_min) + self.t_min
    
    def warp_time(self, u, use_ema=True):
        """Forward transform: u -> t"""
        input_logits = self.input_logits_ema if use_ema else self.input_logits
        output_logits = self.output_logits_ema if use_ema else self.output_logits
        
        input_edges, input_weights = self.get_bin_edges(input_logits)
        output_edges, output_weights = self.get_bin_edges(output_logits)
        
        # Find which bin each u falls into
        bin_idx = torch.searchsorted(input_edges, u.clamp(0, 1))
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)
        
        # Linear interpolation within each bin
        input_left = input_edges[bin_idx]
        input_right = input_edges[bin_idx + 1]
        output_left = output_edges[bin_idx]
        output_right = output_edges[bin_idx + 1]
        
        alpha = (u - input_left) / (input_right - input_left + self.eps)
        warped = output_left + alpha * (output_right - output_left)
        
        return self.denormalize_time(warped)
    
    def get_importance_weights(self, bin_idx=None):
        """Compute importance weights as reciprocal of CDF derivative"""
        _, input_weights = self.get_bin_edges(self.input_logits_ema)
        _, output_weights = self.get_bin_edges(self.output_logits_ema)
        
        # PDF is piecewise constant: ratio of output/input bin sizes
        density = output_weights / (input_weights + self.eps)
        weights = 1.0 / (density + self.eps)
        
        if bin_idx is not None:
            return weights[bin_idx]
        return weights
    
    def get_bin_assignment(self, t):
        """Get bin index for given timesteps"""
        norm_times = self.normalize_time(t)
        input_edges, _ = self.get_bin_edges(self.input_logits_ema)
        bin_idx = torch.searchsorted(input_edges, norm_times.clamp(0, 1))
        return bin_idx.clamp(0, self.num_bins - 1)
    
    def fit_to_losses(self, times, losses, normalize=False):
        """Fit the warping function to observed losses"""
        norm_times = self.normalize_time(times)
        bin_idx = self.get_bin_assignment(times)
        
        # Compute mean loss per bin
        bin_losses = torch.zeros(self.num_bins, device=times.device)
        bin_counts = torch.zeros(self.num_bins, device=times.device)
        bin_losses.index_add_(0, bin_idx, losses)
        bin_counts.index_add_(0, bin_idx, torch.ones_like(losses))
        
        mean_losses = bin_losses / (bin_counts + self.eps)
        
        # Update output logits to match loss distribution
        self.output_logits.data = torch.log(mean_losses + self.eps)
    
    def update_ema(self):
        """Update exponential moving averages of parameters"""
        self.input_logits_ema.data = (
            self.ema_decay * self.input_logits_ema + 
            (1 - self.ema_decay) * self.input_logits.data
        )
        self.output_logits_ema.data = (
            self.ema_decay * self.output_logits_ema + 
            (1 - self.ema_decay) * self.output_logits.data
        )
    
    def get_sampling_timesteps(self, num_steps):
        """Get warped timesteps for sampling"""
        uniform_steps = torch.linspace(0, 1, num_steps, device=self.input_logits.device)
        return self.warp_time(uniform_steps)
    
    def collect_statistics(self, times, losses):
        """Collect loss statistics during epoch"""
        bin_idx = self.get_bin_assignment(times)
        self.epoch_loss_sum.index_add_(0, bin_idx, losses)
        self.epoch_count.index_add_(0, bin_idx, torch.ones_like(losses))
    
    def update_warping(self):
        """Update warping function using collected epoch statistics"""
        mean_losses = self.epoch_loss_sum / (self.epoch_count + self.eps)
        self.output_logits.data = torch.log(mean_losses + self.eps)
        
        # Update EMA
        self.update_ema()
        
        # Reset statistics for next epoch
        self.epoch_loss_sum.zero_()
        self.epoch_count.zero_()