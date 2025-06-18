import os
import torch
import rasterio
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import cv2
import albumentations as A
from augmentations import SentinelAugmentations
from albumentations.pytorch import ToTensorV2 


class PiedmontDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, target_size: Tuple[int, int] = (256, 256), 
                 max_images: int = 3, compute_stats: bool = True,
                 apply_augmentations: bool = False): # Added apply_augmentations flag
        self.root_dir = root_dir
        self.target_size = target_size
        self.max_images = max_images
        self.apply_augmentations = apply_augmentations
        
        self.fire_dirs = self._filter_valid_directories()
        print(f"Found {len(self.fire_dirs)} valid fire directories")
        
        # Initialize the augmentation pipeline
        self.augmentor = SentinelAugmentations()
        
        # For validation/testing, create a minimal transform to just resize and convert to tensor
        self.eval_transform = A.Compose([
            ToTensorV2()
        ])
        self.individual_resize = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1], interpolation=cv2.INTER_LINEAR)
        ])

        if compute_stats:
            self.global_stats = self._compute_global_stats_mean_std()
        else:
            self.global_stats = None
    
    def _filter_valid_directories(self) -> list:
        """Filtra solo le directory che contengono sia file pre-sentinel che GT"""
        valid_dirs = []
        
        all_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) 
                   if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for fire_dir in all_dirs:
            files = os.listdir(fire_dir)
            
        
            sentinel_files = [f for f in files if f.endswith(".tif") and "pre_sentinel" in f]
            
            gt_files = [f for f in files if "GTSentinel" in f]
            
            if len(sentinel_files) > 0 and len(gt_files) > 0:
                valid_dirs.append(fire_dir)
            else:
                missing = []
                if len(sentinel_files) == 0:
                    missing.append("pre-sentinel files")
                if len(gt_files) == 0:
                    missing.append("GT files")
                print(f"Skipping {fire_dir}: missing {', '.join(missing)}")
        
        return valid_dirs
    

    def _compute_global_stats_mean_std(self) -> Dict[str, torch.Tensor]:
        print("Computing global mean and standard deviation for normalization...")
        
        n_bands = 12
        sum_values = torch.zeros(n_bands, dtype=torch.float64)
        sum_sq_values = torch.zeros(n_bands, dtype=torch.float64)
        count_values = torch.zeros(n_bands, dtype=torch.int64)

        
        for fire_dir in self.fire_dirs:
            files = os.listdir(fire_dir)
            sentinel_files = sorted([
                f for f in files 
                if f.endswith(".tif") and "pre_sentinel" in f and "_CM" not in f
            ])
            
            if not sentinel_files:
                continue

            for f_name in sentinel_files: # Note: Not limited by self.max_images here for true global stats
                path = os.path.join(fire_dir, f_name)
                try:
                    with rasterio.open(path) as src:
                        img = src.read() # shape: (12, H, W)
                    
                    # Handle NaNs and apply clamping BEFORE stats calculation for consistency
                    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                    img = np.clip(img, 0, 10000).astype(np.float64) # Use float64 for precision
                    
                    if img.shape[0] != n_bands:
                        print(f"  [DEBUG STATS] Skipping {path}: expected {n_bands} bands, got {img.shape[0]}.")
                        continue 

                    for band_idx in range(n_bands):
                        band_data = img[band_idx].flatten()
                        
                        if band_data.size > 0: 
                            sum_values[band_idx] += np.sum(band_data)
                            sum_sq_values[band_idx] += np.sum(band_data**2)
                            count_values[band_idx] += len(band_data)
                        else: 
                            print(f"[DEBUG STATS] Warning: Band {band_idx} in {path} has no valid data after clipping/nan_to_num for stats.")
                except Exception as e:
                    print(f"  [STATS ERROR] Error processing {path} for stats: {e}")
                    # Continue to next file if one fails, don't break loop

        stats = {
            'mean': torch.zeros(n_bands, dtype=torch.float32),
            'std': torch.zeros(n_bands, dtype=torch.float32)
        }
        
        for band_idx in range(n_bands):
            if count_values[band_idx] > 0:
                mean_val = sum_values[band_idx] / count_values[band_idx]
                variance_val = (sum_sq_values[band_idx] / count_values[band_idx]) - (mean_val**2)
                
                # Correct for potential tiny negative variances due to floating point precision
                if variance_val < 0:
                    variance_val = torch.tensor(0.0, dtype=torch.float64) 

                std_val = torch.sqrt(variance_val)
                
                # Prevent division by zero during normalization if std is extremely small (constant band)
                if std_val < 1e-6: 
                    std_val = torch.tensor(1e-6, dtype=torch.float64) 
                
                stats['mean'][band_idx] = mean_val.to(torch.float32)
                stats['std'][band_idx] = std_val.to(torch.float32)
            else:
                # If a band has no data collected from the entire dataset, set mean=0, std=1
                stats['mean'][band_idx] = 0.0
                stats['std'][band_idx] = 1.0 
                print(f"  [WARNING] No valid data found for band {band_idx} across all samples for stats. Using fallback (0 mean, 1 std).")
            
        print("Global mean and standard deviation computed successfully!")
        print(f"Final Global Means: {stats['mean']}")
        print(f"Final Global STDs: {stats['std']}")
        return stats

    def _normalize_bands_global(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a multi-channel image tensor where channels are stacked
        repetitions of 12 Sentinel bands (e.g., 36 channels for 3 time steps).
        Uses 12 global mean/std and repeats them for each block of 12 channels.
        Expected input image shape: (C_total, H, W), where C_total is a multiple of 12.
        """
        if self.global_stats is None:
            raise ValueError("Global statistics not computed. Cannot normalize.")

        # global_stats['mean'] and ['std'] will be (12,)
        base_mean = self.global_stats['mean'].to(image.device) 
        base_std = self.global_stats['std'].to(image.device)   
        
        total_channels = image.shape[0] # e.g., 36
        num_12_band_blocks = total_channels // 12
        
        # Repeat the 12-element mean/std for each block of 12 channels
        # If total_channels = 36, num_12_band_blocks = 3
        # expanded_mean will be (36,) = [m1..m12, m1..m12, m1..m12]
        expanded_mean = base_mean.repeat(num_12_band_blocks).view(-1, 1, 1) # Reshape to (C_total, 1, 1)
        expanded_std = base_std.repeat(num_12_band_blocks).view(-1, 1, 1)   # Reshape to (C_total, 1, 1)
        
        # Clamping to original Sentinel range (0-10000)
        # This is applied here to the processed image before final normalization.
        # This clamping should be consistent with how stats were calculated.
        image = torch.clamp(image, 0.00, 10000.00) 
        
        # Standardizzazione Z-score
        normalized_image = (image - expanded_mean) / expanded_std
        
        return normalized_image
    


    def __len__(self):
        return len(self.fire_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fire_dir = self.fire_dirs[idx]
        files = os.listdir(fire_dir)

        sentinel_files = sorted([
            f for f in files 
            if f.endswith(".tif") and "pre_sentinel" in f and "_CM" not in f
        ])

        if not sentinel_files:
            raise ValueError(f"No pre-sentinel files found in {fire_dir} - this should not happen!")

        img_np_list = [] # List to hold individual (H, W, 12) numpy arrays
        n_bands = 12 # Number of bands per single Sentinel image

        #print(f"[DEBUG __getitem__] Processing {fire_dir}. Target size: {self.target_size}")


        for f_name in sentinel_files[:self.max_images]: # Limiting to self.max_images for actual input
            path = os.path.join(fire_dir, f_name)
            try:
                with rasterio.open(path) as src:
                    img_read = src.read().astype(np.float32) # Read as float32 NumPy (12, H, W)
                
                # Handle NaNs and apply initial clamping on NumPy array
                img_processed_np = np.nan_to_num(img_read, nan=0.0, posinf=0.0, neginf=0.0)
                img_processed_np = np.clip(img_processed_np, 0.0, 10000.0) 

                img_hwc = np.transpose(img_processed_np, (1, 2, 0)) # -> (H_orig, W_orig, 12)
                
                
                #print(f"  [DEBUG __getitem__] Original HWC shape for {f_name}: {img_hwc.shape}")

                # NEW: Resize individual image to target_size BEFORE stacking
                transformed_individual = self.individual_resize(image=img_hwc)
                resized_img = transformed_individual['image'] # Returns HWC
                
                #print(f"  [DEBUG __getitem__] Resized HWC shape for {f_name}: {resized_img.shape}")
                
                img_np_list.append(resized_img) 
                
            except Exception as e:
                print(f"  [GETITEM ERROR] Error processing {path}: {e}. Appending zero array.")
                H, W = self.target_size
                # Add a zero NumPy array in case of error, ensuring (H, W, C) shape
                img_np_list.append(np.zeros((H, W, n_bands), dtype=np.float32))

        # Padding if less than max_images to ensure consistent stacked input size
        num_missing = self.max_images - len(img_np_list)
        if num_missing > 0:
            # IMPORTANT: For padding, use the actual target_size, not `first_img_shape`.
            # If `img_np_list` is empty, it's (256, 256, 12).
            # If `img_np_list` has items, they *should* already be (256, 256, 12) due to individual_resize.
            # Using first_img_shape here could propagate an error if the first image also failed to resize correctly.
            pad_H, pad_W = self.target_size
            padding_shape = (pad_H, pad_W, n_bands)
            #print(f"  [DEBUG __getitem__] Padding {num_missing} images with shape {padding_shape}")

            for _ in range(num_missing):
                img_np_list.append(np.zeros(padding_shape, dtype=np.float32))


        # Debug: Print shapes of all elements in img_np_list before concatenation
        #for i, img_arr in enumerate(img_np_list):
        #    print(f"  [DEBUG __getitem__] Pre-concat image_np_list[{i}] shape: {img_arr.shape}")

        # Concatenate all NumPy images along the channel dimension
        # Result: (H_orig, W_orig, 12 * max_images)
        stacked_image_np = np.concatenate(img_np_list, axis=-1) 

        # Load Ground Truth mask
        try:
            gt_files = [f for f in files if "GTSentinel" in f]
            if not gt_files:
                raise ValueError(f"No GT file found in {fire_dir} - this should not happen!")
            
            gt_file = gt_files[0]
            gt_path = os.path.join(fire_dir, gt_file)
            
            with rasterio.open(gt_path) as src:
                mask_np = src.read(1) # Read only the first band, (H_orig, W_orig)
            
            # Convert to binary (0 or 1) and ensure uint8 for Albumentations
            mask_np_uint8 = (mask_np > 0).astype(np.uint8)
            #print(f"  [DEBUG __getitem__] Original mask shape: {mask_np_uint8.shape} for {gt_file}")
            resized_mask_data = self.individual_resize(image=mask_np_uint8) # Albumentations expects 'image' for resizing.
            mask_np = resized_mask_data['image'] # This is now (256, 256) uint8

            #print(f"  [DEBUG __getitem__] Resized mask shape: {mask_np.shape}") 

        except Exception as e:
            print(f"  [GETITEM ERROR] Error processing GT for {fire_dir}: {e}")
            # Fallback GT (all non-burned), sized to the current image's spatial dimensions
            mask_np = np.zeros(stacked_image_np.shape[:2], dtype=np.uint8) 


        # --- APPLY AUGMENTATIONS (and Resize) ---
        if self.apply_augmentations:
            # Pass the stacked image (H, W, C_total) and mask (H, W) to the augmentor
            transformed_data = self.augmentor(image=stacked_image_np, mask=mask_np)
            
            processed_image = transformed_data['image'] # Already (C_total, H, W) torch.float32
            processed_mask = transformed_data['mask']   # Already (1, H, W) torch.float32
        else:
            # For validation/test: use a minimal transform with only Resize and ToTensorV2
            transformed_data_eval = self.eval_transform(image=stacked_image_np, mask=mask_np)
            processed_image = transformed_data_eval['image']
            processed_mask = transformed_data_eval['mask'].unsqueeze(0).float() # Ensure mask is (1, H, W) and float32
            
        # --- APPLY GLOBAL NORMALIZATION LAST ---
        # This is the final step to prepare data for the model
        # processed_image has shape (12 * max_images, H, W)
        image_normalized = self._normalize_bands_global(processed_image)
        
        return image_normalized, processed_mask

    def get_sample_info(self, idx: int) -> dict:
        """Restituisce informazioni sul campione per debugging"""
        fire_dir = self.fire_dirs[idx]
        files = os.listdir(fire_dir)
        
        sentinel_files = [f for f in files if f.endswith(".tif") and "pre_sentinel" in f]
        gt_files = [f for f in files if "GTSentinel" in f]
        
        return {
            'fire_dir': fire_dir,
            'sentinel_files': sentinel_files,
            'gt_files': gt_files,
            'num_sentinel': len(sentinel_files),
            'num_gt': len(gt_files)
        }