import torch
import torch.nn.functional as F
import wandb  # For logging metrics
from tqdm import tqdm  # For progress bars
import os
import glob


class TrainingMetrics:
    def __init__(self, run_name):
        self.best_val_loss = float('inf')
        self.run_name = run_name
        
    def update_best_metrics(self, val_loss):
        improved = False
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            improved = True
        return improved

def train_step(model, tokens, mask, optimizer, device):
    optimizer.zero_grad()
    
    t = model.sample_time(tokens.shape[0], tokens.device)
    x0 = model.embedding(tokens)
    noise = model.get_noise(x0, t)
    xt = x0 + noise
    logits = model(xt, mask, t)
    
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        tokens.view(-1),
        ignore_index=0
    )

    if not torch.isnan(loss):
        # Just collect statistics instead of updating
        model.collect_time_statistics(t, loss.detach())
        loss.backward()
        optimizer.step()
    
    return loss.item()

def validation_step(model, tokens, mask, device):
    # Sample time using warping
    t = model.sample_time(tokens.shape[0], tokens.device)
    
    # Get clean embeddings
    x0 = model.embedding(tokens)
    
    # Add noise according to N(0, σt²)
    noise = model.get_noise(x0, t)
    xt = x0 + noise
    
    # Get model predictions
    logits = model(xt, mask, t)
    
    # Compute cross-entropy loss with padding handling
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        tokens.view(-1),
        ignore_index=0  # Assuming 0 is padding token
    )
    
    return loss.item()

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, run_name, is_final=False):
    # Create directories if they don't exist
    save_dir = 'models/final_models' if is_final else 'models/best_models'
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    # Create filename with run name and loss
    filename = f"{run_name}_loss{val_loss:.4f}.pt"
    save_path = os.path.join(save_dir, filename)
    
    if not is_final:
        # Clean up previous best model for this run
        previous_models = glob.glob(os.path.join(save_dir, f"{run_name}_loss*.pt"))
        for old_model in previous_models:
            os.remove(old_model)
    
    torch.save(checkpoint, save_path)
    return save_path

def log_metrics(metrics_dict, step_type='batch'):
    wandb.log(metrics_dict)

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    num_batches = len(train_loader)
    
    # Reset statistics at start of epoch
    model.epoch_loss_history.zero_()
    model.epoch_count_history.zero_()
    
    for batch_idx, (tokens, mask) in enumerate(train_loader):
        tokens = tokens.to(device)
        mask = mask.to(device)
        
        loss = train_step(model, tokens, mask, optimizer, device)
        train_loss += loss
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch {epoch}: [{batch_idx + 1}/{num_batches}] Loss: {loss:.4f}')
            
        log_metrics({
            'train/batch_loss': loss,
            'train/learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
            'batch': batch_idx
        })
    
    # Update time warping at end of epoch
    model.update_time_warping_epoch()
    
    return train_loss / num_batches

def validate_epoch(model, test_loader, device, epoch):
    model.eval()
    val_loss = 0
    num_batches = len(test_loader)
    
    # Collect real sequences
    real_sequences = []
    with torch.no_grad():
        for batch_idx, (tokens, mask) in enumerate(test_loader):
            tokens = tokens.to(device)
            mask = mask.to(device)
            
            loss = validation_step(model, tokens, mask, device)
            val_loss += loss
            
            # Print progress every few batches
            if (batch_idx + 1) % 10 == 0:
                print(f'Validation Epoch {epoch}: [{batch_idx + 1}/{num_batches}] Loss: {loss:.4f}')
            
            real_sequences.extend([seq[seq != 0].cpu().numpy() for seq in tokens])
    
    return val_loss / num_batches



#no model saving
def train_and_validate(model, train_loader, test_loader, optimizer, num_epochs, device, run_name, use_lr_scheduling=True):
    metrics = TrainingMetrics(run_name)
    
    scheduler = None
    if use_lr_scheduling:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
    
    final_val_loss = None
    
    for epoch in range(num_epochs):
        # Training phase
        avg_train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Log to Wandb
        wandb.log({'train/epoch_loss': avg_train_loss, 'epoch': epoch})
        
        # Validation phase (every epoch)
        if epoch % 1 == 0:
            avg_val_loss = validate_epoch(model, test_loader, device, epoch)
            final_val_loss = avg_val_loss
            
            # Log to Wandb
            wandb.log({
                'val/epoch_loss': avg_val_loss,
                'epoch': epoch
            })
            
            if scheduler:
                scheduler.step(avg_val_loss)
            
            if metrics.update_best_metrics(avg_val_loss):
    
                wandb.log({
                    'best_model/val_loss': avg_val_loss,
                    'best_model/train_loss': avg_train_loss,
                    'best_model/epoch': epoch,
                    
                })
    
    return metrics.best_val_loss, final_val_loss