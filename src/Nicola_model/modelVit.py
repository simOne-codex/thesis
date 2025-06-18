########ONLY A START, NEVER TRIED#########################

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm # Import timm library

class MultiModalViTSegmenter(nn.Module): 
    def __init__(self, in_channels_img=72, num_tabular_data=0,
                 vit_model_name='vit_base_patch16_224', # Start with base, or 'vit_small_patch16_224'
                 pretrained=True,
                 img_size=(128, 128),      # Input image size for your dataset
                 patch_size=16,            # Patch size of the chosen ViT
                 decoder_channels=[512, 256, 128, 64], # Channels for the upsampling path
                 out_channels=1):          # Output classes for segmentation (e.g., 1 for binary)
        
        super().__init__()
        self.use_extra = num_tabular_data > 0
        self.img_size = img_size
        self.patch_size = patch_size

        # 1. Vision Transformer Encoder
        self.vit = timm.create_model(
            vit_model_name, 
            pretrained=pretrained, 
            features_only=True, # Get intermediate feature maps from ViT
            in_chans=in_channels_img, # This is where you specify 72 channels
            img_size=img_size         # Important for ViT's positional embeddings
        )
        
        # Determine the output dimensions from the ViT encoder.
        # `features_only=True` returns a list of tensors from different stages.
        # The last tensor in the list (`encoder_features[-1]`) is usually the highest-level feature map.
        # Its channels will be the `embed_dim` of the ViT (e.g., 768 for vit_base).
        encoder_output_dim = self.vit.feature_info.channels()[-1]

        # Calculate the spatial dimensions of the ViT's output feature map
        # If input 128x128, patch_size 16 -> 128/16 = 8. So, 8x8 spatial map
        vit_feature_map_h = img_size[0] // patch_size
        vit_feature_map_w = img_size[1] // patch_size

        # 2. Decoder Part (Upsampling path)
        # This is a simple FPN-like decoder upsampling from the bottleneck.
        # For a full U-Net-like structure, you'd need to extract and connect
        # features from earlier `encoder_features` (e.g., features from different attention blocks).
        
        self.decoder_blocks = nn.ModuleList()
        current_in_channels = encoder_output_dim

        # Number of upsampling steps required to go from (H//patch_size) to img_size
        num_upsample_steps = int(torch.log2(torch.tensor(img_size[0] / vit_feature_map_h)).item())

        # Ensure decoder_channels matches the number of upsampling steps + final conv setup
        # For a simple cascade like this, decoder_channels should typically have `num_upsample_steps` elements
        # E.g., if img_size=128, patch_size=16 => 8x8. Need to go 8->16->32->64->128 (4 upsamples)
        # So, len(decoder_channels) should correspond to the number of upsampling operations.
        # A simple structure could be:
        # decoder_channels = [encoder_output_dim // 2, encoder_output_dim // 4, ...] for each step
        
        # Example decoder structure for 4 upsampling steps (8x8 -> 128x128)
        # Ensure decoder_channels length matches num_upsample_steps if you want this precise flow.
        # Or you can define specific blocks based on expected input/output sizes.
        
        # Let's refine the decoder a bit for typical upsampling progression
        # Starting from 8x8 (128/16) and going to 128x128 requires 4 upsampling steps (8->16->32->64->128)
        # So we need 4 decoder blocks. Let's adjust `decoder_channels` if needed.
        
        # This common pattern will double resolution each step:
        # (B, C, H_in, W_in) -> Conv -> ReLU -> Upsample(x2) -> (B, C_out, H_in*2, W_in*2)
        
        # Adjust decoder_channels if its length doesn't match the required upsampling steps
        # For 128x128 input with 16x16 patches, ViT output is 8x8.
        # To reach 128x128, you need 4 upsampling stages (8->16->32->64->128).
        # So `decoder_channels` should typically have 4 values.
        # e.g., `decoder_channels=[512, 256, 128, 64]` is a common pattern.

        # Initial projection to bridge ViT output to decoder
        # This reshapes the (B, N_patches + 1, Embed_dim) or (B, N_patches, Embed_dim)
        # into (B, Embed_dim, H_spatial, W_spatial) if using `features_only=True`
        # `timm` handles this conversion within `features_only` for the last stage.
        
        for i, out_c in enumerate(decoder_channels):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_in_channels, out_c, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )
            current_in_channels = out_c

        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

        # 3. MLP for tabular data (if any)
        if self.use_extra:
            self.extra_mlp = nn.Sequential(
                nn.Linear(num_tabular_data, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            # Map tabular features to a spatial map.
            # This is a simple example; consider where to fuse (e.g., at a specific decoder stage)
            # For simplicity, mapping to the channel dimension of the final output, then broadcasting.
            self.map_features = nn.Sequential(
                nn.Linear(64, out_channels), # Map to output channels for direct addition
                nn.ReLU() # Use ReLU here, or another activation suitable for fusion
            )
    
    def forward(self, x_img, x_extra=None):
        # Ensure input image size matches ViT's expected input (defined by img_size)
        # Your dataset should ideally handle this, but this is a safeguard.
        if x_img.shape[-2:] != self.img_size:
            x_img = F.interpolate(x_img, size=self.img_size, mode='bilinear', align_corners=False)

        # Encoder pass through ViT
        # `features_only=True` means self.vit(x_img) returns a list of tensors,
        # where each tensor is a feature map from a different stage (attention block).
        encoder_features = self.vit(x_img) 
        x_bottleneck = encoder_features[-1] # Take the highest-level feature map (last one in the list)

        # Decoder pass
        x = x_bottleneck
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x)
            # If you were implementing skip connections:
            # You would take an appropriate feature map from `encoder_features`
            # (e.g., encoder_features[j]) and concatenate it here before the current decoder_block.
            # This requires careful mapping of resolutions and channels.

        seg = self.final_conv(x) 

        # Fusion with tabular data (if any)
        if self.use_extra and x_extra is not None:
            batch_size, _, h, w = seg.shape
            feat = self.extra_mlp(x_extra) 
            feat_map = self.map_features(feat) # (B, out_channels)
            
            # Expand and interpolate the tabular feature to match the segmentation map's spatial dimensions
            feat_map = feat_map.view(batch_size, self.out_channels, 1, 1) # Reshape to (B, C, 1, 1)
            feat_map = F.interpolate(feat_map, size=(h, w), mode='nearest') # Or 'bilinear'

            seg = seg + feat_map # Additive fusion. Consider concatenation if channels permit.

        return seg

'''