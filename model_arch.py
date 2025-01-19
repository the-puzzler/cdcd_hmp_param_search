import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timewarping import TimeWarping

class CategoricalScoreDiffusion(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers=6, num_heads=8, dropout=0.1, 
                 num_piecewise_segments=32, dim_feedforward=128, num_fourier_features=4,
                 t_min=0, t_max=1):
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
        
        # Time warping component (replaces old time warping components)
        self.time_warping = TimeWarping(
            num_bins=num_piecewise_segments,
            t_min=t_min,
            t_max=t_max
        )

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

        
    def get_time_embedding(self, t):
        """Get time embeddings using random Fourier features"""
        t = t.view(-1, 1)  # [batch_size, 1]
        
        # Compute features
        features = t @ self.random_matrix  # [batch_size, num_features]
        
        # Apply sin/cos
        sin_features = torch.sin(features)
        cos_features = torch.cos(features)
        
        # Concatenate
        embeddings = torch.cat([sin_features, cos_features], dim=-1)
        
        # Pass through MLP
        return self.time_mlp(embeddings)
    
    def calculate_score(self, x, expected_x0, t):
        """Compute score: (x0 - x)/t^2"""
        return (expected_x0 - x)/(t[:, None, None]**2)
    
    def get_expected_embedding(self, logits):
        """Convert logits to expected embedding by weighting all possible embeddings"""
        # Exclude padding token (index 0)
        probs = torch.softmax(logits, dim=-1)[:, :, 1:]  # Remove pad token probability
        all_embeddings = self.embedding.get_normalized_weight()[1:]  # Remove pad embedding
        expected_embedding = probs @ all_embeddings
        return expected_embedding

    
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
    
    def forward(self, x, mask, t):
        """
        x: noised embedding state [batch, seq_len, embed_dim]
        mask: attention mask for padding [batch, seq_len]
        t: timestep [batch]
        """
        # Get time embeddings
        t_emb = self.get_time_embedding(t)
        
        # Add time embeddings to input
        x = x + t_emb.unsqueeze(1)  # Broadcasting over seq_len
        
        # Pass through transformer
        # Note: src_key_padding_mask expects True for padding positions
        hidden = self.transformer(x, src_key_padding_mask=mask)
        
        # Project to logits
        logits = self.to_logits(hidden)
        
        # Create attention mask for logits
        mask = mask.unsqueeze(-1).expand(-1, -1, self.vocab_size)
        logits = logits.masked_fill(mask, float('-inf'))
        
        return logits
    
    def sample_time(self, batch_size, device):
        """Sample timesteps using time warping"""
        u = torch.rand(batch_size, device=device)
        return self.time_warping.warp_time(u)
    
    def get_noise(self, embeddings, t):
        """Sample noise n ~ N(0, σt²)"""
        sigma_t = torch.sqrt(t)  # Important: we need sqrt since randn gives N(0,1)
        noise = sigma_t[:, None, None] * torch.randn_like(embeddings)
        return noise

class NormalizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize with σ = 0.001 as per paper, may need to be considered as a hyper parameter
        nn.init.normal_(self.embedding.weight, std=0.001)
    
    def get_normalized_weight(self):
        """Get L2 normalized embedding weights"""
        return F.normalize(self.embedding.weight, p=2, dim=1)
    
    def forward(self, x):
        """Forward pass with L2 normalized embeddings"""
        normalized_weight = self.get_normalized_weight()
        return F.embedding(x, normalized_weight)