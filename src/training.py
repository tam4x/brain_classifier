import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from tqdm.auto import tqdm
import os
from pathlib import Path

def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step scheduler if it's a per-batch scheduler (e.g., OneCycleLR)
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=10, 
    device=None,
    learning_rate=0.001,
    optimizer_type='adam',
    scheduler_type='reduce_on_plateau',
    early_stopping_patience=5,
    save_best_model=True,
    checkpoint_dir='checkpoints',
    weight_decay=1e-4
):
    """
    Train model with advanced training strategies.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
        learning_rate: Initial learning rate
        optimizer_type: 'adam', 'adamw', 'sgd'
        scheduler_type: 'reduce_on_plateau', 'cosine', 'onecycle', None
        early_stopping_patience: Number of epochs to wait before early stopping
        save_best_model: Whether to save the best model checkpoint
        checkpoint_dir: Directory to save checkpoints
        weight_decay: L2 regularization weight
    
    Returns:
        history: Dictionary with training history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print(f"Using optimizer: {optimizer_type}")
    
    # Setup learning rate scheduler
    scheduler = None
    if scheduler_type == 'reduce_on_plateau':
        # Create scheduler with minimal arguments for maximum compatibility
        # Some PyTorch versions don't support 'verbose' parameter
        try:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        except TypeError as e:
            # Fallback: try with even fewer arguments if needed
            if 'verbose' in str(e):
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            else:
                raise
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_type == 'onecycle':
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=learning_rate * 10,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch
        )
    
    if scheduler is not None:
        print(f"Using scheduler: {scheduler_type}")
    
    # Setup checkpoint directory
    if save_best_model:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_path / 'best_model.pth'
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Update learning rate scheduler (for epoch-based schedulers)
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
        
        # Save best model
        if save_best_model and val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history
            }, best_model_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return history

