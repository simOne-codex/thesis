import torch
from utils import fast_hist, fire_area_iou
import numpy as np

def train(model, optimizer, dataloader, loss_fn, device):
    model.train()
    hist = np.zeros((2, 2))  # 2x2 confusion matrix: [TN, FP], [FN, TP]
    total_loss = 0.0
    
    # --- IMPORTANT CHANGE HERE: Unpack all 5 tensors from the dataloader ---
    for batch_idx, (x_pre, pre_mask, x_post, post_mask, y_true) in enumerate(dataloader):
        # --- Move each tensor to the correct device ---
        x_pre = x_pre.to(device)
        pre_mask = pre_mask.to(device)
        x_post = x_post.to(device)
        post_mask = post_mask.to(device)
        y_true = y_true.to(device) # This is your ground truth mask, previously 'targets'
        
        # Forward pass
        # --- IMPORTANT CHANGE HERE: Pass all relevant inputs to the model ---
        outputs = model(x_pre, pre_mask, x_post, post_mask)  # outputs: (B, 1, H, W) - logits
        
        # Calculate loss
        loss = loss_fn(outputs, y_true) # Use y_true for the loss calculation
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate predictions for metrics
        with torch.no_grad():
            # Convert logits to probabilities then to binary predictions
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            # Flatten for confusion matrix calculation
            targets_flat = y_true.cpu().flatten().numpy().astype(int) # Use y_true here
            predicted_flat = predicted.cpu().flatten().numpy().astype(int)
            
            # Update confusion matrix
            hist += fast_hist(targets_flat, predicted_flat, 2)
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"   Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    # Calculate metrics
    iou = fire_area_iou(hist)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, iou

def val(model, dataloader, loss_fn, device):
    model.eval()
    hist = np.zeros((2, 2))  # 2x2 confusion matrix
    total_loss = 0.0
    
    with torch.no_grad():
        # --- IMPORTANT CHANGE HERE: Unpack all 5 tensors from the dataloader ---
        for batch_idx, (x_pre, pre_mask, x_post, post_mask, y_true) in enumerate(dataloader):
            # --- Move each tensor to the correct device ---
            x_pre = x_pre.to(device)
            pre_mask = pre_mask.to(device)
            x_post = x_post.to(device)
            post_mask = post_mask.to(device)
            y_true = y_true.to(device) # This is your ground truth mask
            
            # Forward pass
            # --- IMPORTANT CHANGE HERE: Pass all relevant inputs to the model ---
            outputs = model(x_pre, pre_mask, x_post, post_mask)  # outputs: (B, 1, H, W) - logits
            
            # Calculate loss
            loss = loss_fn(outputs, y_true) # Use y_true for the loss calculation
            total_loss += loss.item()
            
            # Calculate predictions
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            # Flatten for confusion matrix calculation
            targets_flat = y_true.cpu().flatten().numpy().astype(int) # Use y_true here
            predicted_flat = predicted.cpu().flatten().numpy().astype(int)
            
            # Update confusion matrix
            hist += fast_hist(targets_flat, predicted_flat, 2)
    
    # Calculate metrics
    iou = fire_area_iou(hist)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, iou

def predict_and_save(model, dataset, device, save_dir="predictions", num_samples=None):
    """
    Funzione per fare inferenza e salvare le predizioni
    """
    import os
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    model.eval()
    
    # Crea directory se non esiste
    os.makedirs(save_dir, exist_ok=True)
    
    # Se num_samples non specificato, usa tutto il dataset
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    # Usa batch_size=1 per processare un sample alla volta
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            if idx >= num_samples:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)  # logits
            predicted = torch.sigmoid(outputs)  # probabilities
            predicted_binary = (predicted > 0.5).float()  # binary predictions
            
            # Convert to numpy per visualization
            target_np = targets[0, 0].cpu().numpy()  # (H, W)
            pred_prob_np = predicted[0, 0].cpu().numpy()  # (H, W)
            pred_binary_np = predicted_binary[0, 0].cpu().numpy()  # (H, W)
            
            # Get sample info for naming
            sample_info = dataset.get_sample_info(idx)
            fire_name = os.path.basename(sample_info['fire_dir'])
            
            # Create subplot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Ground truth
            axes[0].imshow(target_np, cmap='Reds', vmin=0, vmax=1)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            
            # Predicted probabilities
            im1 = axes[1].imshow(pred_prob_np, cmap='Reds', vmin=0, vmax=1)
            axes[1].set_title('Predicted Probabilities')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Binary prediction
            axes[2].imshow(pred_binary_np, cmap='Reds', vmin=0, vmax=1)
            axes[2].set_title('Binary Prediction (>0.5)')
            axes[2].axis('off')
            
            plt.suptitle(f'Fire: {fire_name}')
            plt.tight_layout()
            
            # Save
            save_path = os.path.join(save_dir, f'{fire_name}_prediction.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved prediction for {fire_name} to {save_path}")
    
    print(f"Predictions saved to {save_dir}/")