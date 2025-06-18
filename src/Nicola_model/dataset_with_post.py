import os
import torch
import rasterio
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import cv2
import albumentations as A
from augmentations import SentinelAugmentations
from albumentations.pytorch import ToTensorV2 
import geopandas as gpd 
import pandas as pd  
from datetime import datetime 
import re 

class PiedmontDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, geojson_path: str, 
                 target_size: Tuple[int, int] = (128, 128), 
                 max_images: int = 6, # Expecting total: e.g., 3 pre and 3 post images
                 compute_stats: bool = True,
                 apply_augmentations: bool = False):
        """
        Initializes the PiedmontDataset.
        Args:
            root_dir (str): Path to the root directory containing fire event folders.
            geojson_path (str): Path to the GeoJSON file containing fire metadata (e.g., 'id', 'initialdate').
            target_size (Tuple[int, int]): Desired spatial resolution (height, width) for output images.
            max_images (int): Maximum *total* number of images to stack across pre and post streams.
                              This should be an even number, with half for pre and half for post.
            compute_stats (bool): Whether to compute global mean/std for normalization or use pre-computed ones.
            apply_augmentations (bool): Whether to apply data augmentations (for training).
        """
        self.root_dir = root_dir
        self.geojson_path = geojson_path
        self.target_size = target_size
        self.max_images = max_images 
        self.num_pre_per_sample = max_images // 2 
        self.num_post_per_sample = max_images // 2 
        self.apply_augmentations = apply_augmentations
        
        if self.max_images % 2 != 0:
            raise ValueError("max_images must be an even number (e.g., 6 for 3 pre + 3 post).")

        self.fire_id_to_date_map = self._load_fire_dates_from_geojson()
        self.fire_dirs = self._filter_valid_directories()
        print(f"Found {len(self.fire_dirs)} valid fire directories for processing.")
        
        # Initialize augmentation pipeline. Now it's the updated SentinelAugmentations.
        self.augmentor = SentinelAugmentations()
        
        # For validation/testing: A minimal transform that includes ONLY ToTensorV2
        # Pixel-level augs are not applied during eval.
        self.eval_transform_to_tensor = A.Compose([
            ToTensorV2() 
        ])

        # Individual resize unchanged.
        self.individual_resize = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1], interpolation=cv2.INTER_LINEAR)
        ])

        if compute_stats:
            self.global_stats = self._compute_global_stats_mean_std()
        else:
            self.global_stats = None
    
    def _load_fire_dates_from_geojson(self) -> Dict[int, datetime]:
        """
        Loads fire IDs and their initial dates from the GeoJSON file using geopandas.
        This map is used to determine if an image is 'pre' or 'post' fire.
        """
        fire_id_to_date = {}
        try:
            gdf = gpd.read_file(self.geojson_path)
            
            if 'id' not in gdf.columns or 'initialdate' not in gdf.columns:
                raise ValueError("GeoJSON file must contain 'id' and 'initialdate' columns.")
            
            # Convert 'initialdate' to datetime objects, coercing errors to NaT
            gdf['parsed_date'] = pd.to_datetime(gdf['initialdate'], errors='coerce')
            
            for index, row in gdf.iterrows():
                fire_id = row['id']
                fire_date_timestamp = row['parsed_date']
                
                if fire_id is not None and pd.notna(fire_date_timestamp):
                    # Ensure fire_id is an integer for lookup
                    fire_id_to_date[int(fire_id)] = fire_date_timestamp.to_pydatetime() 
                else:
                    print(f"Warning: Skipping GeoJSON feature with ID '{fire_id}' due to missing or unparseable initialdate: '{row['initialdate']}'.")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: GeoJSON file not found at {self.geojson_path}. Please check the path.")
        except Exception as e:
            raise RuntimeError(f"Error loading or processing GeoJSON file {self.geojson_path}: {e}")
        
        print(f"Loaded {len(fire_id_to_date)} fire dates from GeoJSON using geopandas.")
        return fire_id_to_date

    def _filter_valid_directories(self) -> list:
        """
        Filters directories to include only those that:
        1. Contain at least one 'pre_sentinel' file.
        2. Contain at least one 'sentinel' (post-fire) file.
        3. Contain a 'GTSentinel' ground truth file.
        4. Their fire_id is present in the loaded GeoJSON metadata.
        """
        valid_dirs = []
        all_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) 
                    if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for fire_dir in all_dirs:
            dir_name = os.path.basename(fire_dir)
            
            # Extract fire ID from directory name (e.g., 'fire_5004' -> 5004)
            match = re.search(r'fire_(\d+)', dir_name)
            fire_id = int(match.group(1)) if match else None

            # Check if fire_id is valid and present in GeoJSON metadata
            if fire_id is None or fire_id not in self.fire_id_to_date_map:
                # print(f"Skipping {fire_dir}: Fire ID {fire_id} not found in GeoJSON or could not be extracted.")
                continue # Silently skip directories without valid fire_id or date

            files = os.listdir(fire_dir)
            
            # Check for presence of required file types
            has_pre_sentinel = any(f.endswith(".tif") and "pre_sentinel" in f for f in files)
            has_post_sentinel = any(f.endswith(".tif") and "sentinel" in f and "pre_sentinel" not in f for f in files)
            has_gt_file = any("GTSentinel" in f for f in files)
            
            if has_pre_sentinel and has_post_sentinel and has_gt_file:
                valid_dirs.append(fire_dir)
            else:
                missing = []
                if not has_pre_sentinel:
                    missing.append("pre-sentinel files")
                if not has_post_sentinel:
                    missing.append("post-sentinel files")
                if not has_gt_file:
                    missing.append("GT files")
                print(f"Skipping {fire_dir}: missing {', '.join(missing)}")
        
        return valid_dirs
    
    def _compute_global_stats_mean_std(self) -> Dict[str, torch.Tensor]:
        """
        Computes global mean and standard deviation for each of the 12 Sentinel-2 bands
        by iterating through all *individual* pre-sentinel AND post-sentinel images in the dataset.
        This provides a consistent normalization for all inputs.
        """
        print("Computing global mean and standard deviation for normalization...")
        
        n_bands = 12 
        sum_values = torch.zeros(n_bands, dtype=torch.float64)
        sum_sq_values = torch.zeros(n_bands, dtype=torch.float64)
        count_values = torch.zeros(n_bands, dtype=torch.int64)

        for fire_dir in self.fire_dirs:
            files = os.listdir(fire_dir)
            
            # Collect all relevant sentinel files (pre and post, excluding cloud masks) for stats computation
            all_sentinel_files_in_dir = sorted([
                f for f in files 
                if f.endswith(".tif") and ("pre_sentinel" in f or "sentinel" in f) and "_CM" not in f
            ])
            
            if not all_sentinel_files_in_dir:
                continue

            for f_name in all_sentinel_files_in_dir:
                path = os.path.join(fire_dir, f_name)
                try:
                    with rasterio.open(path) as src:
                        img = src.read() # Read all bands (12, H, W)
                    
                    # Handle NaNs and apply clamping BEFORE stats calculation for consistency
                    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                    img = np.clip(img, 0, 10000).astype(np.float64) 
                    
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

        stats = {
            'mean': torch.zeros(n_bands, dtype=torch.float32),
            'std': torch.zeros(n_bands, dtype=torch.float32)
        }
        
        for band_idx in range(n_bands):
            if count_values[band_idx] > 0:
                mean_val = sum_values[band_idx] / count_values[band_idx]
                variance_val = (sum_sq_values[band_idx] / count_values[band_idx]) - (mean_val**2)
                
                # Prevent negative variances due to floating point precision
                if variance_val < 0:
                    variance_val = torch.tensor(0.0, dtype=torch.float64) 

                std_val = torch.sqrt(variance_val)
                
                # Prevent division by zero if std is extremely small (constant band)
                if std_val < 1e-6: 
                    std_val = torch.tensor(1e-6, dtype=torch.float64) 
                
                stats['mean'][band_idx] = mean_val.to(torch.float32)
                stats['std'][band_idx] = std_val.to(torch.float32)
            else:
                # Fallback if a band has no valid data across the entire dataset
                stats['mean'][band_idx] = 0.0
                stats['std'][band_idx] = 1.0 
                print(f"  [WARNING] No valid data found for band {band_idx} across all samples for stats. Using fallback (0 mean, 1 std).")
            
        print("Global mean and standard deviation computed successfully!")
        print(f"Final Global Means: {stats['mean']}")
        print(f"Final Global STDs: {stats['std']}")
        return stats

    def _normalize_bands_global(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a multi-channel image tensor using global mean and standard deviation.
        Expected input image shape: (C_total, H, W), where C_total is a multiple of 12.
        The 12-band stats are repeated for each block of 12 channels.
        """
        if self.global_stats is None:
            raise ValueError("Global statistics not computed. Cannot normalize. "
                             "Please compute them or provide them externally.")

        base_mean = self.global_stats['mean'].to(image.device) 
        base_std = self.global_stats['std'].to(image.device)   
        
        total_channels = image.shape[0] 
        num_12_band_blocks = total_channels // 12
        
        # Repeat the 12-element mean/std for each block of 12 channels in the stacked image
        expanded_mean = base_mean.repeat(num_12_band_blocks).view(-1, 1, 1) 
        expanded_std = base_std.repeat(num_12_band_blocks).view(-1, 1, 1)   
        
        # Apply initial clamping to the raw image values before normalization
        image = torch.clamp(image, 0.00, 10000.00) 
        
        # Perform Z-score normalization
        normalized_image = (image - expanded_mean) / expanded_std
        
        return normalized_image
    
    def __len__(self):
        """Returns the total number of valid fire event directories in the dataset."""
        return len(self.fire_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single data sample (stacked Sentinel imagery and GT mask).

        Args:
            idx (int): Index of the fire event directory to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                - pre_fire_images_normalized (torch.Tensor): Stacked and normalized pre-fire imagery (C_pre, H, W).
                - pre_fire_mask (torch.Tensor): Binary mask for pre-fire imagery (C_pre, H, W), 1 for real, 0 for padded.
                - post_fire_images_normalized (torch.Tensor): Stacked and normalized post-fire imagery (C_post, H, W).
                - post_fire_mask (torch.Tensor): Binary mask for post-fire imagery (C_post, H, W), 1 for real, 0 for padded.
                - processed_mask (torch.Tensor): Ground truth burn mask (1, H, W).
        """
        fire_dir = self.fire_dirs[idx]
        dir_name = os.path.basename(fire_dir)
        
        match = re.search(r'fire_(\d+)', dir_name)
        fire_id = int(match.group(1)) if match else None

        if fire_id is None or fire_id not in self.fire_id_to_date_map:
            raise ValueError(f"Internal error: Fire ID {fire_id} for directory {fire_dir} not found in GeoJSON map.")

        fire_event_date = self.fire_id_to_date_map[fire_id]
        files = os.listdir(fire_dir)

        # 1. Collect all potential pre_sentinel and sentinel files
        all_sentinel_candidates = []
        for f_name in files:
            if f_name.endswith(".tif") and "_CM" not in f_name and ("pre_sentinel" in f_name or "sentinel" in f_name):
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', f_name)
                if date_match:
                    file_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                    all_sentinel_candidates.append({'filename': f_name, 'date': file_date, 'path': os.path.join(fire_dir, f_name)})
        
        all_sentinel_candidates.sort(key=lambda x: x['date'])

        # 2. Separate into pre-fire and post-fire lists, handling duplicates
        pre_fire_paths_available = []
        post_fire_paths_available = []
        seen_post_dates = set() 

        for candidate in all_sentinel_candidates:
            if candidate['date'] >= fire_event_date:
                if candidate['date'] not in seen_post_dates:
                    post_fire_paths_available.append(candidate['path'])
                    seen_post_dates.add(candidate['date'])
            elif "pre_sentinel" in candidate['filename']:
                pre_fire_paths_available.append(candidate['path'])
        
        # 3. Select the desired number of pre and post images
        selected_pre_images = pre_fire_paths_available[-self.num_pre_per_sample:] 
        selected_post_images = post_fire_paths_available[:self.num_post_per_sample] 

        pre_img_np_list = []
        pre_mask_np_list = []
        post_img_np_list = []
        post_mask_np_list = []
        n_bands = 12 
        H, W = self.target_size
        padding_img_shape = (H, W, n_bands)
        padding_mask_shape = (H, W, n_bands) 

        # 4. Process and pad Pre-Fire Images
        for path in selected_pre_images:
            try:
                with rasterio.open(path) as src:
                    img_read = src.read().astype(np.float32) 
                img_processed_np = np.nan_to_num(img_read, nan=0.0, posinf=0.0, neginf=0.0)
                img_processed_np = np.clip(img_processed_np, 0.0, 10000.0) 
                img_hwc = np.transpose(img_processed_np, (1, 2, 0)) 
                transformed_individual = self.individual_resize(image=img_hwc)
                resized_img = transformed_individual['image'] # Returns HWC
                pre_img_np_list.append(resized_img) 
                pre_mask_np_list.append(np.ones(padding_mask_shape, dtype=np.float32)) 
            except Exception as e:
                print(f"  [GETITEM ERROR] Error processing pre-fire {path}: {e}. Appending zero array and zero mask.")
                pre_img_np_list.append(np.zeros(padding_img_shape, dtype=np.float32))
                pre_mask_np_list.append(np.zeros(padding_mask_shape, dtype=np.float32)) 

        num_missing_pre = self.num_pre_per_sample - len(pre_img_np_list)
        for _ in range(num_missing_pre):
            pre_img_np_list.append(np.zeros(padding_img_shape, dtype=np.float32))
            pre_mask_np_list.append(np.zeros(padding_mask_shape, dtype=np.float32))

        # 5. Process and pad Post-Fire Images
        for path in selected_post_images:
            try:
                with rasterio.open(path) as src:
                    img_read = src.read().astype(np.float32) 
                img_processed_np = np.nan_to_num(img_read, nan=0.0, posinf=0.0, neginf=0.0)
                img_processed_np = np.clip(img_processed_np, 0.0, 10000.0) 
                img_hwc = np.transpose(img_processed_np, (1, 2, 0)) 
                transformed_individual = self.individual_resize(image=img_hwc)
                resized_img = transformed_individual['image'] 
                post_img_np_list.append(resized_img) 
                post_mask_np_list.append(np.ones(padding_mask_shape, dtype=np.float32)) 
            except Exception as e:
                print(f"  [GETITEM ERROR] Error processing post-fire {path}: {e}. Appending zero array and zero mask.")
                post_img_np_list.append(np.zeros(padding_img_shape, dtype=np.float32))
                post_mask_np_list.append(np.zeros(padding_mask_shape, dtype=np.float32)) 

        num_missing_post = self.num_post_per_sample - len(post_img_np_list)
        for _ in range(num_missing_post):
            post_img_np_list.append(np.zeros(padding_img_shape, dtype=np.float32))
            post_mask_np_list.append(np.zeros(padding_mask_shape, dtype=np.float32))

        # Concatenate all NumPy images and masks along the channel dimension
        stacked_pre_image_np = np.concatenate(pre_img_np_list, axis=-1) 
        stacked_pre_mask_np = np.concatenate(pre_mask_np_list, axis=-1) 

        stacked_post_image_np = np.concatenate(post_img_np_list, axis=-1) 
        stacked_post_mask_np = np.concatenate(post_mask_np_list, axis=-1) 

        # 6. Load and preprocess Ground Truth mask
        try:
            gt_files = [f for f in files if "GTSentinel" in f]
            if not gt_files:
                raise ValueError(f"No GT file found in {fire_dir} - this should not happen as it's filtered!")
            
            gt_file = gt_files[0]
            gt_path = os.path.join(fire_dir, gt_file)
            
            with rasterio.open(gt_path) as src:
                mask_np_orig = src.read(1) 
            
            mask_np = (mask_np_orig > 0).astype(np.uint8) # Ensure GT mask is uint8 for albumentations
            resized_mask_data = self.individual_resize(image=mask_np) 
            gt_mask_np = resized_mask_data['image'] # This is now (H, W) uint8

        except Exception as e:
            print(f"  [GETITEM ERROR] Error processing GT for {fire_dir}: {e}. Appending zero mask.")
            gt_mask_np = np.zeros(self.target_size, dtype=np.uint8) 

        # 7. Apply Augmentations (if enabled) or minimal evaluation transform
        if self.apply_augmentations:
            augmented_data = self.augmentor(
                pre_img_np=stacked_pre_image_np,
                post_img_np=stacked_post_image_np,
                pre_mask_np=stacked_pre_mask_np,
                post_mask_np=stacked_post_mask_np,
                gt_mask_np=gt_mask_np
            )
            pre_images_tensor = augmented_data['pre_img']
            pre_masks_tensor = augmented_data['pre_mask']
            post_images_tensor = augmented_data['post_img']
            post_masks_tensor = augmented_data['post_mask']
            processed_mask = augmented_data['gt_mask']

        else:
            # For evaluation, use the self.eval_transform_to_tensor
            # Pixel-level augmentations are skipped here.
            pre_images_tensor = self.eval_transform_to_tensor(image=stacked_pre_image_np)['image']
            pre_masks_tensor = self.eval_transform_to_tensor(image=stacked_pre_mask_np)['image']
            post_images_tensor = self.eval_transform_to_tensor(image=stacked_post_image_np)['image']
            post_masks_tensor = self.eval_transform_to_tensor(image=stacked_post_mask_np)['image']
            processed_mask = self.eval_transform_to_tensor(image=gt_mask_np)['image'].float() # Ensure (1,H,W) float

        # 8. Apply global normalization as the final preprocessing step to the IMAGE data
        pre_images_normalized = self._normalize_bands_global(pre_images_tensor)
        post_images_normalized = self._normalize_bands_global(post_images_tensor)

        # Padding masks are already float32 from SentinelAugmentations __call__ return
        
        return pre_images_normalized, pre_masks_tensor, \
               post_images_normalized, post_masks_tensor, \
               processed_mask

    def get_sample_info(self, idx: int) -> dict:
        """
        Returns information about a specific sample for debugging purposes.
        """
        fire_dir = self.fire_dirs[idx]
        files = os.listdir(fire_dir)
        
        sentinel_pre_files = [f for f in files if f.endswith(".tif") and "pre_sentinel" in f]
        sentinel_post_files = [f for f in files if f.endswith(".tif") and "sentinel" in f and "pre_sentinel" not in f]
        gt_files = [f for f in files if "GTSentinel" in f]
        
        dir_name = os.path.basename(fire_dir)
        match = re.search(r'fire_(\d+)', dir_name)
        fire_id = int(match.group(1)) if match else None
        fire_date = self.fire_id_to_date_map.get(fire_id, "N/A")

        return {
            'fire_dir': fire_dir,
            'fire_id': fire_id,
            'fire_event_date': fire_date.strftime('%Y-%m-%d') if isinstance(fire_date, datetime) else str(fire_date),
            'pre_sentinel_files': sentinel_pre_files,
            'post_sentinel_files': sentinel_post_files,
            'gt_files': gt_files,
            'num_pre_sentinel': len(sentinel_pre_files),
            'num_post_sentinel': len(sentinel_post_files),
            'num_gt': len(gt_files)
        }