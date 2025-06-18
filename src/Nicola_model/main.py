import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from train import train, val
from utils import poly_lr_scheduler
from dataset_with_post import PiedmontDataset  
from model import MultiModalUNet
import segmentation_models_pytorch as smp

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
torch.autograd.set_detect_anomaly(True)

# Define hyperparameters
BATCH_SIZE = 8
EPOCHS = 150
target_size = (128, 128)
max_images = 6 
lr = 1e-4
WARMUP_EPOCHS = 5 # Number of epochs for the warm-up phase
WARMUP_LR_START = 1e-6 # Starting LR for warm-up (e.g., 1/100th of base LR)


# Loss function binary
bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True).to(device)
BCE_WEIGHT=0.2
DICE_WEIGHT=0.8
def combined_loss_fn(outputs, masks):
    return BCE_WEIGHT*bce_loss_fn(outputs, masks) + DICE_WEIGHT*dice_loss_fn(outputs, masks)

# Model
model = MultiModalUNet(in_channels_img_per_stream=36, num_tabular_data=0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
warmup_scheduler = LinearLR(optimizer, 
                            start_factor=WARMUP_LR_START / lr, # Ratio of starting LR to base LR
                            total_iters=WARMUP_EPOCHS)

# Main Scheduler (CosineAnnealingWarmRestarts)
main_scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                            T_0=50, 
                                            T_mult=1, 
                                            eta_min=1e-6)

 
#The scheduler will first use warmup_scheduler for WARMUP_EPOCHS,
#and then switch to main_scheduler for the remaining epochs.
scheduler = SequentialLR(optimizer, 
                         schedulers=[warmup_scheduler, main_scheduler], 
                         milestones=[WARMUP_EPOCHS]) 
# --- CORRECTED DATASET AND DATALOADER SETUP ---

# 1. Create a "dummy" dataset instance *just* to compute global statistics.
#    Augmentations are set to False here as they are not relevant for stats computation.
#    This dataset will be used to get the global_stats and fire_dirs list.
print("Initializing dataset to compute global statistics and prepare data splits...")
stats_dataset = PiedmontDataset(
    root_dir="piedmont", 
    target_size=target_size, 
    max_images=max_images,
    geojson_path="/nfs/home/genovese/thesis-wildfire-genovese/data/nicola/piedmont_2012_2024_fa.geojson",
    compute_stats=True, # This will trigger the global stats computation
    apply_augmentations=False # No augmentations needed for stats computation
)

# Extract global stats and the filtered list of fire directories
global_stats_mean = stats_dataset.global_stats['mean']
global_stats_std = stats_dataset.global_stats['std']
all_fire_dirs = stats_dataset.fire_dirs # This is the complete, filtered list of directories

print(f"Total dataset size (after filtering): {len(all_fire_dirs)}")

# 2. Split the *indices* of the fire directories
train_size = int(0.8 * len(all_fire_dirs))
val_size = len(all_fire_dirs) - train_size
train_indices, val_indices = random_split(range(len(all_fire_dirs)), [train_size, val_size])

print(f"Train dataset size: {len(train_indices)}, Validation dataset size: {len(val_indices)}")

# 3. Create actual training and validation dataset instances
#    Each will get its own set of `fire_dirs` and `apply_augmentations` flag.

# Training Dataset: apply augmentations
train_dataset = PiedmontDataset(
    root_dir="piedmont",
    geojson_path="/nfs/home/genovese/thesis-wildfire-genovese/data/nicola/piedmont_2012_2024_fa.geojson",
    target_size=target_size, # Or whatever size you chose
    max_images=6, # Example: 3 pre + 3 post
    compute_stats=False, # Or False if you're loading pre-computed stats
    apply_augmentations=True
)
# Assign the pre-computed global stats and the specific fire directories for this split
train_dataset.global_stats = {'mean': global_stats_mean, 'std': global_stats_std}
train_dataset.fire_dirs = [all_fire_dirs[i] for i in train_indices]


# Validation Dataset: NO augmentations
val_dataset = PiedmontDataset(
    root_dir="piedmont",
    geojson_path="/nfs/home/genovese/thesis-wildfire-genovese/data/nicola/piedmont_2012_2024_fa.geojson",
    target_size=target_size, # Or whatever size you chose
    max_images=6, # Example: 3 pre + 3 post
    compute_stats=False, # Or False if you're loading pre-computed stats
    apply_augmentations=False
)
# Assign the pre-computed global stats and the specific fire directories for this split
val_dataset.global_stats = {'mean': global_stats_mean, 'std': global_stats_std}
val_dataset.fire_dirs = [all_fire_dirs[i] for i in val_indices]


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- END CORRECTED SETUP ---

# Training history
train_losses = []
train_ious = []
val_losses = []
val_ious = []

print("Starting training...")
start_epoch = 0

#comment
'''
checkpoint = torch.load('model_aug_corrected.pth',weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#lr = optimizer.param_groups[0]['lr']
start_epoch = checkpoint['epoch'] + 1
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print("Start epoch:", start_epoch)
'''

for epoch in range(start_epoch, EPOCHS):
    #current_lr = poly_lr_scheduler(optimizer, init_lr=lr, iter=(epoch-60), max_iter=EPOCHS)
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    #print(f"Current Learning Rate: {current_lr:.6f}")
    
    # Training
    train_loss, train_iou = train(model, optimizer=optimizer, dataloader=train_loader, 
                                  loss_fn=combined_loss_fn, device=device)
    
    # Validation
    val_loss, val_iou = val(model, dataloader=val_loader, loss_fn=combined_loss_fn, device=device)
    
    # Store metrics
    train_losses.append(train_loss)
    train_ious.append(train_iou)
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    if val_iou==max(val_ious):
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_iou': val_iou,
        'val_loss': val_loss,
        }, 'best_model_150.pth')
    
    print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou,
        'val_loss': val_loss,
        }, 'model_aug_150.pth')
    
    scheduler.step()

print("Training completed!")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_ious, label='Train IoU')
plt.plot(val_ious, label='Val IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('Training and Validation IoU')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

print(f"Best validation IoU: {max(val_ious):.4f}")