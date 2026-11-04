import torch
import os
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """
    Save the model and optimizer state.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load the model and optimizer state.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
