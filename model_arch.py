import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CategoricalScoreDiffusion(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers=6, num_heads=8, dropout=0.1, 
                 num_piecewise_segments=32, dim_feedforward=128, num_fourier_features=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Normalized embedding layer
        self.embedding = NormalizedEmbedding(vocab_size, embed_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.to_logits = nn.Linear(embed_dim, vocab_size)
        
        # Time warping components
        self.num_segments = num_piecewise_segments
        self.register_buffer('time_weights', torch.ones(num_piecewise_segments))
        self.register_buffer('loss_history', torch.zeros(num_piecewise_segments))
        self.register_buffer('count_history', torch.zeros(num_piecewise_segments))

        # Random Fourier Features for time embedding
        self.register_buffer('random_matrix', 
            torch.randn(1, num_fourier_features) * 2 * np.pi)  # For sin/cos encoding
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * num_fourier_features, embed_dim),  # 2x because of sin/cos
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
    
    def get_fourier_features(self, t):
        """Convert time to random Fourier features"""
        t_proj = t.unsqueeze(-1) @ self.random_matrix  # Project time to higher dimension
        fourier_features = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        return fourier_features
        
    def get_time_embedding(self, t):
        """Get full time embedding"""
        fourier_features = self.get_fourier_features(t)
        return self.time_mlp(fourier_features)
    
    def calculate_score(self, x, expected_x0, t):
        """Compute score: (x0 - x)/t^2"""
        return (expected_x0 - x)/(t[:, None, None]**2)
    
    def get_expected_embedding(self, logits):
        """Convert logits to expected embedding by weighting all possible embeddings"""
        # Exclude padding token (index 0)
        probs = torch.softmax(logits, dim=-1)[:, :, 1:]  # Remove pad token probability
        all_embeddings = self.embedding.get_normalized_weight()[1:]  # Remove pad embedding
        expected_embedding = torch.einsum('bsv,ve->bse', probs, all_embeddings)
        return expected_embedding
    
    def sample_time(self, batch_size, device):
        """Sample timesteps using current time warping"""
        u = torch.rand(batch_size, device=device)
        return self.warp_time(u)
    
    def get_noise(self, embeddings, t):
        """
        Sample noise n ~ N(0, σt²)
        embeddings: [batch, seq_len, embed_dim]
        t: [batch]
        """
        # σt = t  (standard deviation is sqrt of variance)
        sigma_t = torch.sqrt(t)  # Important: we need sqrt since randn gives N(0,1)
        
        # Sample n ~ N(0, σt²) by scaling standard normal
        noise = sigma_t[:, None, None] * torch.randn_like(embeddings)
        
        return noise
    
    def warp_time(self, u):
        normalized_weights = F.softmax(self.time_weights, dim=0)
        cumsum = torch.cumsum(normalized_weights, dim=0)
   
        
        # Clamp u to avoid edge cases
        u = torch.clamp(u, 0.0, 0.9999)
 
        # Get segment indices and clamp them
        segment_idx = torch.searchsorted(cumsum, u)
        segment_idx = torch.clamp(segment_idx, 0, self.num_segments - 1)
    
        # Linear interpolation within segment
        start_t = segment_idx.float() / self.num_segments
        end_t = (segment_idx.float() + 1) / self.num_segments
        
        # Get previous cumsum values
        prev_cumsum = torch.zeros_like(u)
        prev_cumsum = torch.where(segment_idx > 0, 
                                cumsum[segment_idx - 1], 
                                prev_cumsum)
        
        segment_u = (u - prev_cumsum) / normalized_weights[segment_idx]
        t = start_t + segment_u * (end_t - start_t)
        
        result = torch.clamp(t, 0.0, 1.0)
    
        
        return result
    
    def update_time_warping(self, t, loss):
        """Update time warping statistics based on observed loss"""
        segment_idx = (t * self.num_segments).long().clamp(0, self.num_segments-1)
        # Expand loss to match batch size
        batch_losses = torch.full_like(t, loss.item())
        self.loss_history.index_add_(0, segment_idx, batch_losses)
        self.count_history.index_add_(0, segment_idx, torch.ones_like(t))
        
        # Update weights to make loss more uniform
        avg_loss = self.loss_history / (self.count_history + 1e-8)
        self.time_weights.copy_(torch.log(avg_loss + 1e-8))
    
    def forward(self, x, mask, t):
        """
        x: noised embedding state [batch, seq_len, embed_dim]
        mask: attention mask for padding [batch, seq_len]
        t: timestep [batch]
        """
        t_emb = self.get_time_embedding(t)
        x = x + t_emb.unsqueeze(1)
        hidden = self.transformer(x, src_key_padding_mask=mask)
        logits = self.to_logits(hidden)
        
        # Create attention mask for logits
        mask = mask.unsqueeze(-1).expand(-1, -1, self.vocab_size)
        logits = logits.masked_fill(mask, float('-inf'))
        
        return logits

class NormalizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    
    def get_normalized_weight(self):
        """Get L2 normalized embedding weights"""
        return F.normalize(self.embedding.weight, p=2, dim=1)
    
    def forward(self, x):
        """Forward pass with L2 normalized embeddings"""
        normalized_weight = self.get_normalized_weight()
        return F.embedding(x, normalized_weight)