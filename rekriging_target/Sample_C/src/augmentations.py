import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from typing import Tuple, Dict
import cv2 # Required for A.BORDER_REFLECT_101 if using ShiftScaleRotate

class SentinelAugmentations:
    def __init__(self, p_spatial=0.5,p_flip=0.5, p_color=0.5):
        # 1. Define SPATIAL augmentations (apply to all images and masks)
        self.spatial_transform = A.Compose([
            A.HorizontalFlip(p=p_flip),
            A.VerticalFlip(p=p_flip),
            A.RandomRotate90(p=p_flip),
            # A.ShiftScaleRotate( # If uncommented, ensure cv2 is imported and correct border_mode
            #     shift_limit=0.05,
            #     scale_limit=0.05,
            #     rotate_limit=15,
            #     p=p_transform,
            #     border_mode=cv2.BORDER_REFLECT_101
            # ),
        ], p=p_spatial,
        # Crucial: Define additional targets.
        # We will pass 'pre_img_np' as the primary 'image' and 'gt_mask_np' as the primary 'mask'.
        # The other inputs will be passed as keyword arguments and matched by these additional_targets.
        additional_targets={
            'post_img_input': 'image', # This will be the stacked_post_image_np
            'pre_mask_input': 'image', # This will be the stacked_pre_mask_np (binary 0/1, multi-channel)
            'post_mask_input': 'image', # This will be the stacked_post_mask_np (binary 0/1, multi-channel)
            # 'gt_mask_input' is implicitly handled as the primary 'mask' argument
            # 'pre_img_input' is implicitly handled as the primary 'image' argument
        })

        # 2. Define PIXEL-LEVEL augmentations (apply only to the *image* data, not masks)
        self.pixel_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        ],p=p_color)
        

        # 3. Define TO_TENSOR conversion (apply to each item separately after all augs)
        self.to_tensor_transform = ToTensorV2()

    def __call__(self,
                 pre_img_np: np.ndarray,
                 post_img_np: np.ndarray,
                 pre_mask_np: np.ndarray,
                 post_mask_np: np.ndarray,
                 gt_mask_np: np.ndarray) -> Dict[str, torch.Tensor]:

        # 1. Apply SPATIAL transformations to all inputs consistently
        # IMPORTANT: Pass one image as the 'image' argument, and the GT mask as the 'mask' argument.
        # The rest are passed as keyword arguments which match the 'additional_targets'.
        transformed_spatial = self.spatial_transform(
            image=pre_img_np,                      # This is the primary 'image' Albumentations uses for shape etc.
            mask=gt_mask_np.astype(np.uint8),      # This is the primary 'mask' Albumentations uses.
            post_img_input=post_img_np,            # This matches 'post_img_input' in additional_targets
            pre_mask_input=pre_mask_np,            # This matches 'pre_mask_input' in additional_targets
            post_mask_input=post_mask_np,          # This matches 'post_mask_input' in additional_targets
        )

        # Retrieve spatially augmented data
        # Note: The keys for retrieving are the ones actually passed to the A.Compose call.
        pre_img_aug_spatial = transformed_spatial['image']       # Retrieved using the 'image' key
        gt_mask_aug_spatial = transformed_spatial['mask']        # Retrieved using the 'mask' key
        post_img_aug_spatial = transformed_spatial['post_img_input']
        pre_mask_aug_spatial = transformed_spatial['pre_mask_input']
        post_mask_aug_spatial = transformed_spatial['post_mask_input']

        # 2. Apply PIXEL-LEVEL transformations ONLY to the image data (pre and post)
        pre_img_aug_pixel = self.pixel_transform(image=pre_img_aug_spatial)['image']
        post_img_aug_pixel = self.pixel_transform(image=post_img_aug_spatial)['image']

        # 3. Convert all (image and mask) to PyTorch Tensors (CHW)
        pre_img_tensor = self.to_tensor_transform(image=pre_img_aug_pixel)['image']
        post_img_tensor = self.to_tensor_transform(image=post_img_aug_pixel)['image']

        pre_mask_tensor = self.to_tensor_transform(image=pre_mask_aug_spatial)['image']
        post_mask_tensor = self.to_tensor_transform(image=post_mask_aug_spatial)['image']
        gt_mask_tensor = self.to_tensor_transform(image=gt_mask_aug_spatial)['image']

        # Ensure GT mask is (1, H, W) and float32
        #gt_mask_tensor = gt_mask_tensor.unsqueeze(0) if gt_mask_tensor.ndim == 2 else gt_mask_tensor
        gt_mask_tensor = gt_mask_tensor.float()

        # Ensure padding masks are float32
        pre_mask_tensor = pre_mask_tensor.float()
        post_mask_tensor = post_mask_tensor.float()

        return {
            'pre_img': pre_img_tensor,
            'post_img': post_img_tensor,
            'pre_mask': pre_mask_tensor,
            'post_mask': post_mask_tensor,
            'gt_mask': gt_mask_tensor
        }