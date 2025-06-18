import torch
import torch.nn as nn
import torch.nn.functional as F # Import F for interpolation
import segmentation_models_pytorch as smp


class MultiModalUNet(nn.Module):
    def __init__(self, in_channels_img_per_stream=36, num_tabular_data=0):
        """
        Initializes the MultiModalUNet.

        Args:
            in_channels_img_per_stream (int): Number of input channels for each image stream (pre-fire and post-fire).
                                              E.g., 36 if using 3 images * 12 bands per stream.
            num_tabular_data (int): Number of features in the tabular data. Set to 0 if not used.
        """
        super().__init__()
        self.use_extra = num_tabular_data > 0
        
        # --- Image Encoders (Pre-fire and Post-fire) ---
        # Define two separate encoders. We use the smp.Unet constructor and extract its encoder part.
        # This is because smp.Unet provides a convenient way to get an encoder with pre-trained weights
        # and a compatible decoder structure.
        
        # Encoder for pre-fire imagery
        self.pre_encoder_net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels_img_per_stream, 
            classes=1, # Dummy: not used directly for the final output of this 'net'
            activation=None # Keep None for raw logits
        )

        # Encoder for post-fire imagery
        # Using "imagenet" weights for both is a common starting point. You could try None for post-fire
        # if you suspect pre-trained weights might hinder learning for post-event changes.
        self.post_encoder_net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet", 
            in_channels=in_channels_img_per_stream, 
            classes=1, # Dummy
            activation=None
        )

        # --- Shared Decoder and Segmentation Head ---
        # The decoder and segmentation head will process the combined features from both encoders.
        # We can take the decoder and head components from one of the UNet instances
        # because their internal structure (based on 'resnet34') is compatible.
        self.decoder = self.pre_encoder_net.decoder
        self.segmentation_head = self.pre_encoder_net.segmentation_head

        # --- MLP for Tabular Data (Optional) ---
        # This part remains the same as your original implementation.
        if self.use_extra:
            self.extra_mlp = nn.Sequential(
                nn.Linear(num_tabular_data, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            # Map tabular features to a spatial map
            self.map_features = nn.Sequential(
                nn.Linear(64, 256), # Output size should match desired feature map dimensions (e.g., 16x16=256)
                nn.ReLU()
            )
    
    def forward(self, x_pre: torch.Tensor, pre_mask: torch.Tensor, 
                x_post: torch.Tensor, post_mask: torch.Tensor, 
                x_extra: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the MultiModalUNet.

        Args:
            x_pre (torch.Tensor): Pre-fire image stack (Batch, C_pre, H, W).
            pre_mask (torch.Tensor): Mask for pre-fire input (Batch, C_pre, H, W), 1 for real data, 0 for padded.
            x_post (torch.Tensor): Post-fire image stack (Batch, C_post, H, W).
            post_mask (torch.Tensor): Mask for post-fire input (Batch, C_post, H, W), 1 for real data, 0 for padded.
            x_extra (torch.Tensor, optional): Tabular data (Batch, num_tabular_data). Defaults to None.

        Returns:
            torch.Tensor: Predicted segmentation mask (Batch, 1, H, W).
        """
        # 1. Process pre-fire and post-fire imagery through their respective encoders
        # .encoder() returns a list of feature maps from different stages of the encoder.
        # Example for ResNet34: [x_0, x_1, x_2, x_3, x_4, x_5] where x_0 is the original input.
        pre_features = self.pre_encoder_net.encoder(x_pre)
        post_features = self.post_encoder_net.encoder(x_post)

        # 2. Derive spatial masks from the input padding masks
        # The input masks (pre_mask, post_mask) are (B, C_stream, H, W).
        # We sum along the channel dimension to get a single spatial mask (B, 1, H, W)
        # indicating where original input pixels were truly present (non-padded).
        # `keepdim=True` maintains the channel dimension for easier interpolation.
        spatial_pre_mask = (pre_mask.sum(dim=1, keepdim=True) > 0).float() # (B, 1, H, W)
        spatial_post_mask = (post_mask.sum(dim=1, keepdim=True) > 0).float() # (B, 1, H, W)

        # 3. Fuse feature maps from both encoders, applying padding masks
        decoder_input_features = []
        
        # The first element of encoder_features (index 0) is typically the raw input.
        # The decoder uses this for the initial skip connection. We'll use the pre-fire raw input.
        decoder_input_features.append(pre_features[0]) 

        # Loop through the feature maps from subsequent encoder stages (from index 1 to last)
        for i in range(1, len(pre_features)): 
            pre_f = pre_features[i]  # Feature map from pre-encoder at stage 'i'
            post_f = post_features[i] # Feature map from post-encoder at stage 'i'
            
            # Get current spatial dimensions of the feature map
            current_h, current_w = pre_f.shape[2], pre_f.shape[3]
            
            # Resample the spatial masks to match the current feature map's dimensions
            # Using 'nearest' interpolation mode is good for binary masks to preserve 0/1 values.
            resampled_pre_mask = F.interpolate(spatial_pre_mask, size=(current_h, current_w), mode='nearest')
            resampled_post_mask = F.interpolate(spatial_post_mask, size=(current_h, current_w), mode='nearest')
            
            # Apply masks: Element-wise multiplication to zero out features derived from padded regions.
            # This explicitly tells the model to ignore features where no original data was present.
            masked_pre_f = pre_f * resampled_pre_mask
            masked_post_f = post_f * resampled_post_mask
            
            # Fuse features: Simple element-wise summation.
            # This adds the masked feature maps. Summation is effective for similar modalities
            # and maintains the original channel dimensions, which allows direct use of smp.Unet's decoder.
            fused_feature_level = masked_pre_f + masked_post_f
            
            # Add the fused feature map to the list for the decoder
            decoder_input_features.append(fused_feature_level)
            
        # 4. Pass the fused (and masked) features through the shared decoder
        seg_features = self.decoder(decoder_input_features)
        
        # 5. Apply the final segmentation head
        seg = self.segmentation_head(seg_features)

        # 6. Fuse with tabular data (if enabled)
        if self.use_extra and x_extra is not None:
            batch_size, _, h, w = seg.shape
            feat = self.extra_mlp(x_extra)          # (B, 64)
            feat_map = self.map_features(feat)      # (B, 256)
            
            # Reshape tabular features into a small spatial map (e.g., 1x16x16 for 256 features)
            # Assuming 256 output from map_features maps to a 16x16 map with 1 channel.
            # Adjust view dimensions if map_features output different sizes or channels.
            feat_map = feat_map.view(batch_size, 1, 16, 16) 
            
            # Interpolate the tabular feature map to match the segmentation output size
            feat_map = F.interpolate(feat_map, size=(h, w), mode='bilinear', align_corners=False)
            
            # Add the tabular feature map to the segmentation output (fusion at output-level)
            seg = seg + feat_map 

        return seg