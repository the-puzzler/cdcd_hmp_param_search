import torch
import torch.nn.functional as F

def ce_loss_simple(logits, target_tokens, temperature=0.1): #avg over seq then batch
    B, S, V = logits.shape
    logits_flat = logits.view(-1, V) / temperature
    targets_flat = target_tokens.view(-1)
    
    # Create mask for non-pad tokens
    mask = (targets_flat != 0).view(B, S)
    
    # Calculate CE loss per token
    token_losses = F.cross_entropy(
        logits_flat, 
        targets_flat, 
        reduction='none'
    ).view(B, S)
    
    # Average over sequence first (using mask)
    seq_lengths = mask.sum(dim=1)
    sequence_loss = (token_losses * mask).sum(dim=1) / seq_lengths
    
    # Average over batch
    loss = sequence_loss.mean()
    
    return loss