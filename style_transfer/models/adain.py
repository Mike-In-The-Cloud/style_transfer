"""
Implements the AdaIN (Adaptive Instance Normalization) style transfer model.

This module contains the core components for the AdaIN style transfer technique,
including the AdaIN layer, VGG-based encoder, a corresponding decoder, and the main
StyleTransferModel that orchestrates the process.
"""
import torch
import torch.nn as nn
# import torch.nn.functional as F # F is not used, can be removed
from torchvision.models import VGG19_Weights
from typing import Tuple, Optional # Tuple is not used, can be removed
import logging
import os # Added for path operations

# Assuming image_utils.py is in style_transfer.utils
from ..utils.image_processing import save_tensor_as_image # Relative import for use within package

logger = logging.getLogger(__name__)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization (AdaIN) layer.

    This layer aligns the mean and variance of the content features with those
    of the style features.
    """
    def __init__(self, eps: float = 1e-5):
        """Initialize the AdaIN layer.

        Args:
            eps: A small epsilon value to prevent division by zero during normalization.
        """
        super().__init__()
        self.eps = eps

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Apply Adaptive Instance Normalization.

        The content features are normalized and then scaled and shifted using the
        mean and standard deviation of the style features.

        Args:
            content (torch.Tensor): Content feature tensor of shape (N, C, H, W).
            style (torch.Tensor): Style feature tensor of shape (N, C, H', W').
                                   Spatial dimensions can be different from content.

        Returns:
            torch.Tensor: The stylized feature tensor, with the same shape as content.
        """
        # Validate input dimensions
        assert len(content.shape) == 4, "Content tensor must be 4D (N, C, H, W)"
        assert len(style.shape) == 4, "Style tensor must be 4D (N, C, H', W')"

        size = content.size()
        # Calculate style statistics (mean and std) across spatial dimensions
        style_mean = style.mean(dim=[2, 3], keepdim=True)
        style_std = style.std(dim=[2, 3], keepdim=True) + self.eps
        logger.info(f"AdaIN - Style Mean: shape={style_mean.shape}, mean={style_mean.mean().item():.4f}, min={style_mean.min().item():.4f}, max={style_mean.max().item():.4f}")
        logger.info(f"AdaIN - Style Std: shape={style_std.shape}, mean={style_std.mean().item():.4f}, min={style_std.min().item():.4f}, max={style_std.max().item():.4f}")

        # Calculate content statistics (mean and std) across spatial dimensions
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True) + self.eps
        logger.info(f"AdaIN - Content Mean: shape={content_mean.shape}, mean={content_mean.mean().item():.4f}, min={content_mean.min().item():.4f}, max={content_mean.max().item():.4f}")
        logger.info(f"AdaIN - Content Std: shape={content_std.shape}, mean={content_std.mean().item():.4f}, min={content_std.min().item():.4f}, max={content_std.max().item():.4f}")

        # Normalize content features
        normalized_content = (content - content_mean) / content_std
        # Apply style statistics to normalized content
        return normalized_content * style_std + style_mean

class Encoder(nn.Module):
    """VGG19-based encoder for extracting content and style features.

    This encoder uses the pre-trained VGG19 model from torchvision,
    specifically its feature extraction layers up to 'relu4_1'.
    The weights of the encoder are frozen during training/inference.
    """
    def __init__(self):
        super().__init__()
        # Define the VGG19 architecture layers EXACTLY as in naoto0804/pytorch-AdaIN/net.py
        # This is the 'vgg' sequential model from their net.py
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(3, 64, (3, 3)),\
            nn.ReLU(),  # relu1-1 (layer index 3)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(64, 64, (3, 3)),\
            nn.ReLU(),  # relu1-2 (layer index 6)\
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(64, 128, (3, 3)),\
            nn.ReLU(),  # relu2-1 (layer index 10)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(128, 128, (3, 3)),\
            nn.ReLU(),  # relu2-2 (layer index 13)\
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(128, 256, (3, 3)),\
            nn.ReLU(),  # relu3-1 (layer index 17)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(256, 256, (3, 3)),\
            nn.ReLU(),  # relu3-2 (layer index 20)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(256, 256, (3, 3)),\
            nn.ReLU(),  # relu3-3 (layer index 23)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(256, 256, (3, 3)),\
            nn.ReLU(),  # relu3-4 (layer index 26)\
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(256, 512, (3, 3)),\
            nn.ReLU(),  # relu4-1 (layer index 30) <--- This is our target output for AdaIN features\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(512, 512, (3, 3)),\
            nn.ReLU(),  # relu4-2 (layer index 33)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(512, 512, (3, 3)),\
            nn.ReLU(),  # relu4-3 (layer index 36)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(512, 512, (3, 3)),\
            nn.ReLU(),  # relu4-4 (layer index 39)\
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(512, 512, (3, 3)),\
            nn.ReLU(),  # relu5-1 (layer index 43)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(512, 512, (3, 3)),\
            nn.ReLU(),  # relu5-2 (layer index 46)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(512, 512, (3, 3)),\
            nn.ReLU(),  # relu5-3 (layer index 49)\
            nn.ReflectionPad2d((1, 1, 1, 1)),\
            nn.Conv2d(512, 512, (3, 3)),\
            nn.ReLU()  # relu5-4 (layer index 52)\
        )

        # Store the index of relu4_1 (0-indexed within self.features)
        self.relu4_1_idx = 30

        # Freeze parameters of the encoder as it's used only for feature extraction
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features from the input image tensor up to relu4_1.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Feature tensor from relu4_1 layer.
        """
        # Pass through layers sequentially and tap output at relu4_1_idx
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.relu4_1_idx:
                return x
        # Should not be reached if relu4_1_idx is correctly defined and within bounds
        logger.warning("Encoder.forward: Did not find relu4_1_idx during feature pass. Returning full pass.") # pragma: no cover
        return x # pragma: no cover

class Decoder(nn.Sequential):
    """Decoder network to reconstruct an image from stylized features.

    The architecture typically mirrors the encoder, using convolutional layers
    and upsampling to generate an image from the feature map provided by AdaIN.
    This class inherits from nn.Sequential to directly match the structure of
    the pre-trained weights from naoto0804/pytorch-AdaIN.
    """
    def __init__(self):
        super().__init__(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # Block 1
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), # End of block similar to VGG's conv4_x
            # Block 2
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # Block 3
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Block 4
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # Block 5
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Output layer
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0),
        )
        # No forward method needed as nn.Sequential handles it.

class StyleTransferModel(nn.Module):
    """Complete AdaIN-based Style Transfer Model.

    This model combines an Encoder (VGG19-based) to extract features,
    an AdaIN layer to merge content and style features, and a Decoder
    to reconstruct the stylized image.
    """
    def __init__(self, vgg_weights_path: str = 'models/vgg_normalised.pth', decoder_weights_path: str = 'models/decoder.pth'):
        """Initializes the Encoder, Decoder, and AdaIN components.
           Loads pre-trained weights for Encoder and Decoder.
        """
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.adain = AdaIN()

        try:
            self.encoder.features.load_state_dict(torch.load(vgg_weights_path))
            logger.info(f"Successfully loaded VGG weights from {vgg_weights_path}")
        except FileNotFoundError:
            logger.error(f"VGG weights file not found at {vgg_weights_path}. Ensure it's downloaded and in the correct path.")
            raise
        except RuntimeError as e:
            logger.error(f"Error loading VGG weights from {vgg_weights_path}: {e}. Check model architecture compatibility.")
            raise

        try:
            self.decoder.load_state_dict(torch.load(decoder_weights_path))
            logger.info(f"Successfully loaded Decoder weights from {decoder_weights_path}")
        except FileNotFoundError:
            logger.error(f"Decoder weights file not found at {decoder_weights_path}. Ensure it's downloaded and in the correct path.")
            raise
        except RuntimeError as e:
            logger.error(f"Error loading Decoder weights from {decoder_weights_path}: {e}. Check model architecture compatibility.")
            raise

        # Freeze encoder parameters again after loading, just in case
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Decoder parameters should remain trainable if we were to fine-tune, but for inference, it's not strictly necessary to set requires_grad.

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features from an input image tensor using the encoder.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Feature tensor.
        """
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Generates an image from a feature tensor using the decoder.

        Args:
            x (torch.Tensor): Feature tensor.

        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        return self.decoder(x)

    def forward(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Performs the core style transfer operation.

        The process involves:
        1. Extracting features from both content and style images using the encoder.
        2. Applying Adaptive Instance Normalization (AdaIN) to align content features
           with style features.
        3. Interpolating between the AdaIN-stylized features and original content features
           using the style strength `alpha`.
        4. Decoding the resulting features to produce the final stylized image.

        Args:
            content (torch.Tensor): Content image tensor (N, 3, H, W).
            style (torch.Tensor): Style image tensor (N, 3, H', W').
            alpha (float, optional): Style strength. A value of 1.0 applies full styling,
                                     while 0.0 retains the original content features.
                                     Defaults to 1.0.

        Returns:
            torch.Tensor: Stylized image tensor (N, 3, H_out, W_out).
        """
        # Step 1: Extract features
        content_features = self.encode(content)
        style_features = self.encode(style)

        # Step 2: Apply AdaIN to transfer style statistics
        stylized_features = self.adain(content_features, style_features)

        # Step 3: Interpolate between stylized and content features based on alpha
        # alpha = 1 means full style, alpha = 0 means full content
        interpolated_features = alpha * stylized_features + (1 - alpha) * content_features

        # Step 4: Decode the features to generate the output image
        output_image = self.decode(interpolated_features)
        return output_image

    def transfer_style(self, content: torch.Tensor, style: torch.Tensor,
                      alpha: float = 1.0, preserve_color: bool = False) -> torch.Tensor:
        """
        High-level function to transfer style from a style image to a content image,
        with an option to preserve the original colors of the content image.

        Args:
            content (torch.Tensor): Content image tensor (N, 3, H, W), expected range [0, 1].
            style (torch.Tensor): Style image tensor (N, 3, H', W'), expected range [0, 1].
            alpha (float, optional): Style strength, controlling the interpolation between
                                     content and stylized features. Defaults to 1.0.
            preserve_color (bool, optional): If True, the color characteristics of the
                                           original content image are preserved in the
                                           stylized output. Defaults to False.

        Returns:
            torch.Tensor: Stylized image tensor, with pixel values typically in [-1, 1]
                          due to the decoder's Tanh, but color preservation step might clamp to [0,1].
                          The ImageProcessor.postprocess_image should handle final normalization.
        """
        if preserve_color:
            # Color preservation strategy: Perform style transfer on a grayscale version of the style image.
            # Then, match the luminance of the stylized output to the original content image's luminance.

            # Convert style image to grayscale. Standard NTSC conversion weights.
            # Ensure style_gray has 3 channels for compatibility with the encoder.
            style_gray = 0.299 * style[:, 0:1, :, :] + \
                         0.587 * style[:, 1:2, :, :] + \
                         0.114 * style[:, 2:3, :, :]
            style_gray = style_gray.repeat(1, 3, 1, 1) # Repeat grayscale channel to form 3-channel image

            # Perform style transfer using the grayscale style image
            # The `forward` method handles feature extraction, AdaIN, alpha blending, and decoding.
            stylized_output_with_gray_style = self.forward(content, style_gray, alpha)

            # Convert content image to grayscale (luminance)
            content_lum = 0.299 * content[:, 0:1, :, :] + \
                          0.587 * content[:, 1:2, :, :] + \
                          0.114 * content[:, 2:3, :, :]

            # Convert the stylized output (from gray style) to grayscale (luminance)
            output_lum = 0.299 * stylized_output_with_gray_style[:, 0:1, :, :] + \
                         0.587 * stylized_output_with_gray_style[:, 1:2, :, :] + \
                         0.114 * stylized_output_with_gray_style[:, 2:3, :, :]

            # Adjust the colors of the stylized output to match the content's luminance profile.
            # This is a common technique for color preservation in style transfer.
            # Adding a small epsilon to output_lum to prevent division by zero.
            color_adjusted_output = stylized_output_with_gray_style * (content_lum / (output_lum + 1e-6))

            # Clamp values to a valid image range [0, 1] as this operation can push values outside.
            # The VGG normalization expects [0,1] range primarily, and Tanh outputs [-1,1].
            # ImageProcessor handles final conversion, but clamping here is safer for color preservation.
            return torch.clamp(color_adjusted_output, 0, 1)

        # If not preserving color, perform standard style transfer.
        return self.forward(content, style, alpha)

    def generate_alpha_sequence_frames(self, content_tensor, style_tensor,
                                       num_frames: int, output_dir_path: str,
                                       file_prefix="frame_"):
        logger.info(f"Generating AdaIN alpha sequence ({num_frames} frames) into {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True)

        content_f = self.encoder(content_tensor)
        style_f = None
        base_stylized_features = None

        # Only compute style-related features if they are actually needed for stylization
        if num_frames > 0: # and at least one frame will have alpha > 0
            style_f = self.encoder(style_tensor)
            base_stylized_features = self.adain(content_f, style_f)

        if num_frames == 0:
            logger.warning("generate_alpha_sequence_frames called with num_frames=0. No frames will be generated.")
            return

        for i in range(num_frames):
            # Determine alpha for the current frame
            if num_frames == 1:
                current_alpha = 1.0 # Single frame usually means full style for a sequence context
            else:
                current_alpha = i / (num_frames - 1)

            # Interpolate features
            if current_alpha == 0 or base_stylized_features is None: # Pure content or style features unavailable
                final_features = content_f
            elif current_alpha == 1.0:
                final_features = base_stylized_features
            else:
                final_features = current_alpha * base_stylized_features + (1 - current_alpha) * content_f

            decoded_frame = self.decoder(final_features)
            frame_filename = f"{file_prefix}{i:03d}.png"
            frame_save_path = os.path.join(output_dir_path, frame_filename)

            # save_tensor_as_image expects a single image tensor (C, H, W) and not a batch.
            # The output of decoder (g_t) is typically (1, C, H, W) if batch size is 1.
            save_tensor_as_image(decoded_frame.cpu().detach().squeeze(0), frame_save_path)

        logger.info(f"Finished generating {num_frames} frames in {output_dir_path}")

# Placeholder for VGG and Decoder if not defined in this file
# This is just to make the class structure runnable for thought process
if not hasattr(nn, 'Sequential'): # Simple check if this is run standalone without PyTorch context
    class VGGEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Linear(10,10) # Dummy
        def forward(self, x):
            return self.features(x)

    class DecoderNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Linear(10,10) # Dummy
        def forward(self, x):
            return self.layers(x)

    # Replace with actual VGG encoder and Decoder model definitions used in the project
    # vgg_encoder = VGGEncoder()
    # adain_decoder = DecoderNet()