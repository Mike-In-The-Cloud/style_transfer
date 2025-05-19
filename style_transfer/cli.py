import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch

from style_transfer import APP_NAME, __version__
from style_transfer.models.adain import StyleTransferModel
from style_transfer.models.johnson_transformer import JohnsonNet
from style_transfer.utils.device_utils import get_device, clear_gpu_memory
from style_transfer.utils.image_processing import (
    load_image_as_tensor,
    save_tensor_as_image,
    get_image_size
)
from style_transfer.utils.model_utils import load_johnson_model
from style_transfer.utils.video_utils import create_gif_from_frames, generate_interpolated_frames

# Configure logger for the module
logger = logging.getLogger(__name__)

def restricted_float(x):
    """Helper type for argparse to restrict float to 0.0-1.0 range."""
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} not a floating-point literal")
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} not in range [0.0, 1.0]")
    return x

def run_style_transfer_pipeline(
    content_path_str: str,
    style_path_str: Optional[str],
    output_path_str: str,
    model_type: str,
    alpha: float = 1.0,
    preserve_color: bool = False,
    johnson_model_weights: Optional[str] = None,
    max_size: int = 512,
    device_name: Optional[str] = None,
    generate_adain_alpha_sequence: bool = False,
    adain_sequence_num_frames: int = 20,
    adain_sequence_gif_path: Optional[str] = None,
    adain_gif_frame_duration: float = 0.1,
    generate_johnson_gif: bool = False,
    johnson_gif_frames: int = 20,
    johnson_gif_duration: float = 0.1,
    johnson_gif_path: Optional[str] = None,
    johnson_gif_style_intensity: float = 1.0,
    johnson_output_blend_alpha: float = 1.0,
    adain_gif_ping_pong: bool = False,
    johnson_gif_ping_pong: bool = False
):
    """
    Main pipeline to run style transfer.
    Handles AdaIN (with optional GIF sequence) and Johnson models (with optional GIF sequence).
    """
    device = get_device(device_name)
    logger.info(f"Using device: {device}")

    content_path = Path(content_path_str)
    style_path = Path(style_path_str) if style_path_str else None
    output_path = Path(output_path_str)

    if not content_path.exists():
        logger.error(f"Content image not found: {content_path}")
        raise FileNotFoundError(f"Content image not found: {content_path}")

    if model_type == "adain" and (not style_path or not style_path.exists()):
        logger.error(f"Style image not found or not provided, but required for AdaIN: {style_path}")
        raise FileNotFoundError(f"Style image not found or not provided, but required for AdaIN: {style_path}")

    if output_path.parent:
        os.makedirs(output_path.parent, exist_ok=True)

    logger.info("Loading content image...")
    content_tensor = load_image_as_tensor(content_path, max_size=max_size, model_type=model_type).to(device)

    style_tensor = None
    if model_type == "adain" and style_path:
        logger.info("Loading style image for AdaIN...")
        style_tensor = load_image_as_tensor(style_path, max_size=max_size, model_type=model_type).to(device)
    elif model_type == "adain" and style_tensor is None:
        err_msg = "Style tensor is None for AdaIN model despite passing initial checks. This is an internal error."
        logger.error(err_msg)
        raise ValueError(err_msg)

    stylized_output_tensor = None
    adain_model_instance = None
    johnson_model_instance = None

    try:
        if model_type == "adain":
            logger.info("Initializing AdaIN StyleTransferModel...")
            project_root = Path(__file__).resolve().parent.parent
            vgg_weights = str(project_root / "models" / "vgg_normalised.pth")
            decoder_weights = str(project_root / "models" / "decoder.pth")

            adain_model_instance = StyleTransferModel(vgg_weights_path=vgg_weights,
                                                  decoder_weights_path=decoder_weights).to(device)
            adain_model_instance.eval()

            logger.info("Running AdaIN style transfer for main image...")
            stylized_output_tensor = adain_model_instance.transfer_style(
                content_tensor, style_tensor,
                alpha=alpha, preserve_color=preserve_color
            )

            if generate_adain_alpha_sequence:
                if style_tensor is None:
                    logger.warning("Cannot generate AdaIN alpha sequence: style_tensor is None. Skipping sequence generation.")
                elif not adain_sequence_gif_path:
                    logger.warning("generate_adain_alpha_sequence is True, but adain_sequence_gif_path is not provided. Skipping GIF creation.")
                else:
                    logger.info(f"Starting AdaIN alpha sequence generation ({adain_sequence_num_frames} frames) for GIF: {adain_sequence_gif_path}")
                    gif_output_dir = Path(adain_sequence_gif_path).parent
                    if gif_output_dir:
                        os.makedirs(gif_output_dir, exist_ok=True)

                    with tempfile.TemporaryDirectory(prefix=f"{APP_NAME}_adain_frames_") as tmp_frames_dir:
                        logger.info(f"Saving AdaIN sequence frames to temporary directory: {tmp_frames_dir}")
                        try:
                            adain_model_instance.generate_alpha_sequence_frames(
                                content_tensor, style_tensor,
                                num_frames=adain_sequence_num_frames,
                                output_dir_path=tmp_frames_dir
                            )
                            logger.info(f"Creating GIF from frames at {tmp_frames_dir} to {adain_sequence_gif_path}")
                            if not create_gif_from_frames(
                                tmp_frames_dir,
                                adain_sequence_gif_path,
                                frame_duration=adain_gif_frame_duration,
                                ping_pong=adain_gif_ping_pong
                            ):
                                logger.error(f"Failed to create AdaIN sequence GIF at {adain_sequence_gif_path}")
                            else:
                                logger.info(f"Successfully created AdaIN sequence GIF: {adain_sequence_gif_path}")
                        except Exception as e_seq:
                            logger.error(f"Error during AdaIN sequence generation or GIF creation: {e_seq}", exc_info=True)
        elif model_type == "johnson":
            logger.info("Initializing Johnson Transformer model...")
            if not johnson_model_weights or not Path(johnson_model_weights).exists():
                err_msg = f"Johnson model weights not found or not specified: {johnson_model_weights}"
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)

            johnson_network = load_johnson_model(johnson_model_weights, device)
            johnson_model_instance = JohnsonNet(model=johnson_network).to(device)
            johnson_model_instance.eval()

            logger.info("Running Johnson style transfer...")
            # stylized_output_tensor is raw output from TransformerNetwork (no Tanh), likely [0,255] range
            stylized_output_tensor = johnson_model_instance(content_tensor)

            # Blend the final Johnson output with content image if alpha < 1.0
            if johnson_output_blend_alpha < 0.999: # Use < 0.999 to handle potential float inaccuracies
                logger.info(f"Blending Johnson output with content image using alpha: {johnson_output_blend_alpha}")
                # Ensure both tensors are on the same device and squeezed if necessary for broadcasting
                # content_tensor is already [0,255] and on the correct device.
                # stylized_output_tensor is also [0,255] and on the correct device.
                # The tensors should already have compatible shapes (e.g., [1, C, H, W] or [C, H, W])
                # Let's bring them to CPU for blending to be safe, then move back if needed (though save_tensor_as_image handles CPU tensors)

                content_cpu = content_tensor.cpu()
                stylized_cpu = stylized_output_tensor.cpu()

                # Perform blending
                blended_tensor = (1.0 - johnson_output_blend_alpha) * content_cpu + \
                                 johnson_output_blend_alpha * stylized_cpu

                # Clamp just in case, though linear interpolation of [0,255] images should stay in range
                blended_tensor = torch.clamp(blended_tensor, 0, 255)
                stylized_output_tensor = blended_tensor.to(device) # Move back to original device if further processing needed, though save_tensor_as_image takes CPU

            if generate_johnson_gif:
                if not johnson_gif_path:
                    logger.warning("generate_johnson_gif is True, but johnson_gif_path is not provided. Skipping GIF creation.")
                elif stylized_output_tensor is None:
                    logger.warning("Cannot generate Johnson GIF: stylized_output_tensor is None after model run. Skipping GIF generation.")
                else:
                    logger.info(f"Starting Johnson interpolated sequence generation ({johnson_gif_frames} frames) for GIF: {johnson_gif_path}")
                    gif_output_dir = Path(johnson_gif_path).parent
                    if gif_output_dir:
                        os.makedirs(gif_output_dir, exist_ok=True)

                    with tempfile.TemporaryDirectory(prefix=f"{APP_NAME}_johnson_frames_") as tmp_frames_dir:
                        logger.info(f"Saving Johnson sequence frames to temporary directory: {tmp_frames_dir}")
                        try:
                            generate_interpolated_frames(
                                content_tensor_orig=content_tensor,
                                stylized_tensor_orig=stylized_output_tensor,
                                num_frames=johnson_gif_frames,
                                output_dir_path=tmp_frames_dir,
                                gif_style_intensity=johnson_gif_style_intensity
                            )
                            logger.info(f"Creating GIF from Johnson frames at {tmp_frames_dir} to {johnson_gif_path}")
                            if not create_gif_from_frames(
                                tmp_frames_dir,
                                johnson_gif_path,
                                frame_duration=johnson_gif_duration,
                                ping_pong=johnson_gif_ping_pong
                            ):
                                logger.error(f"Failed to create Johnson sequence GIF at {johnson_gif_path}")
                            else:
                                logger.info(f"Successfully created Johnson sequence GIF: {johnson_gif_path}")
                        except Exception as e_seq:
                            logger.error(f"Error during Johnson sequence generation or GIF creation: {e_seq}", exc_info=True)

        else:
            unknown_model_msg = f"Unknown model_type: {model_type}"
            logger.error(unknown_model_msg)
            raise ValueError(unknown_model_msg)

        if stylized_output_tensor is not None:
            logger.info(f"Saving main stylized image to: {output_path}")
            save_tensor_as_image(stylized_output_tensor.cpu().detach().squeeze(0), str(output_path), model_type=model_type)
            logger.info("Main style transfer complete and image saved.")
        else:
            no_output_msg = "Stylized output tensor is None. Main image not saved."
            logger.error(no_output_msg)
            raise RuntimeError(no_output_msg)

    except Exception as e:
        logger.error(f"Error during style transfer pipeline: {e}", exc_info=True)
        raise
    finally:
        if device.type != 'cpu':
            logger.info("Clearing GPU memory...")
            if adain_model_instance: del adain_model_instance
            if johnson_model_instance: del johnson_model_instance
            clear_gpu_memory()
            torch.cuda.empty_cache()

def main():
    """Parses command-line arguments and executes the style transfer pipeline."""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - Neural Style Transfer CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--content", type=str, required=True, help="Path to the content image.")
    parser.add_argument("--output", type=str, default="styled_output.png", help="Path to save the stylized output image.")
    parser.add_argument("--style", type=str, help="Path to the style image (required for AdaIN model type).")

    parser.add_argument("--model_type", type=str, default="adain", choices=["adain", "johnson"],
                        help="Type of style transfer model to use.")
    parser.add_argument("--johnson_model_weights", type=str,
                        help="Path to pre-trained weights for Johnson model (required if model_type is 'johnson').")

    parser.add_argument("--alpha", type=restricted_float, default=1.0,
                        help="Alpha for AdaIN style strength (0.0 to 1.0).")
    parser.add_argument("--preserve_color", action="store_true",
                        help="Preserve content image color (AdaIN model_type only).")

    parser.add_argument("--generate_adain_gif", action="store_true",
                        help="Generate a GIF of AdaIN alpha transition (AdaIN model_type only).")
    parser.add_argument("--adain_gif_frames", type=int, default=20,
                        help="Number of frames for AdaIN GIF.")
    parser.add_argument("--adain_gif_path", type=str, default=None,
                        help="Output path for the AdaIN alpha sequence GIF. Defaults to a temp file if not set but generation is enabled.")
    parser.add_argument("--adain_gif_duration", type=float, default=0.1,
                        help="Duration (in seconds) of each frame in the AdaIN GIF.")
    parser.add_argument("--adain_gif_ping_pong", action="store_true",
                        help="Enable ping-pong effect for AdaIN GIF (plays forwards then backwards).")

    parser.add_argument("--generate_johnson_gif", action="store_true",
                        help="Generate a GIF of Johnson style interpolation (Johnson model_type only).")
    parser.add_argument("--johnson_gif_frames", type=int, default=20,
                        help="Number of frames for Johnson GIF.")
    parser.add_argument("--johnson_gif_duration", type=float, default=0.1,
                        help="Duration (in seconds) of each frame in the Johnson GIF.")
    parser.add_argument("--johnson_gif_path", type=str, default=None,
                        help="Output path for the Johnson interpolated GIF. Defaults to a temp file if not set but generation is enabled.")
    parser.add_argument("--johnson_gif_style_intensity", type=restricted_float, default=1.0,
                        help="Controls the style intensity at the end of the Johnson GIF (0.0=content, 1.0=full style).")
    parser.add_argument("--johnson_output_blend_alpha", type=restricted_float, default=1.0,
                        help="Alpha to blend the final Johnson output image with the content image (0.0=content, 1.0=stylized).")
    parser.add_argument("--johnson_gif_ping_pong", action="store_true",
                        help="Enable ping-pong effect for Johnson GIF (plays forwards then backwards).")

    parser.add_argument("--max_size", type=int, default=512,
                        help="Maximum dimension (width or height) to which images will be resized.")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu', 'mps', 'auto'], default='auto',
                        help="Compute device to use (e.g., 'cuda', 'cpu', 'mps'). 'auto' will attempt to use GPU if available.")

    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.debug("Verbose logging enabled.")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.model_type == "johnson" and not args.johnson_model_weights:
        parser.error("--johnson_model_weights is required when --model_type is 'johnson'.")
    if args.model_type == "adain" and not args.style:
        parser.error("--style image path is required when --model_type is 'adain'.")

    actual_device_name = args.device if args.device != 'auto' else None

    derived_adain_gif_path = None
    if args.generate_adain_gif:
        if args.model_type == "adain":
            output_p = Path(args.output)
            gif_name = f"{output_p.stem}_alpha_sequence.gif"
            derived_adain_gif_path = str(output_p.parent / gif_name)
            logger.info(f"AdaIN GIF, if generated, will be saved to: {derived_adain_gif_path}")
        else:
            logger.warning("--generate_adain_gif is specified, but model_type is not 'adain'. AdaIN GIF will not be generated.")
            args.generate_adain_gif = False

    derived_johnson_gif_path = None
    if args.generate_johnson_gif:
        if args.model_type == "johnson":
            output_p = Path(args.output)
            gif_name = f"{output_p.stem}_interpolated_sequence.gif"
            derived_johnson_gif_path = str(output_p.parent / gif_name)
            logger.info(f"Johnson GIF, if generated, will be saved to: {derived_johnson_gif_path}")
        else:
            logger.warning("--generate_johnson_gif is specified, but model_type is not 'johnson'. Johnson GIF will not be generated.")
            args.generate_johnson_gif = False

    try:
        run_style_transfer_pipeline(
            content_path_str=args.content,
            style_path_str=args.style,
            output_path_str=args.output,
            model_type=args.model_type,
            alpha=args.alpha,
            preserve_color=args.preserve_color,
            johnson_model_weights=args.johnson_model_weights,
            max_size=args.max_size,
            device_name=actual_device_name,
            generate_adain_alpha_sequence=args.generate_adain_gif,
            adain_sequence_num_frames=args.adain_gif_frames,
            adain_sequence_gif_path=derived_adain_gif_path,
            adain_gif_frame_duration=args.adain_gif_duration,
            generate_johnson_gif=args.generate_johnson_gif,
            johnson_gif_frames=args.johnson_gif_frames,
            johnson_gif_duration=args.johnson_gif_duration,
            johnson_gif_path=derived_johnson_gif_path,
            johnson_gif_style_intensity=args.johnson_gif_style_intensity,
            johnson_output_blend_alpha=args.johnson_output_blend_alpha,
            adain_gif_ping_pong=args.adain_gif_ping_pong,
            johnson_gif_ping_pong=args.johnson_gif_ping_pong
        )
        logger.info(f"Processing finished successfully. Output: {args.output}")
        if derived_adain_gif_path and args.generate_adain_gif:
             logger.info(f"AdaIN GIF processing also attempted. Check: {derived_adain_gif_path}")
        if derived_johnson_gif_path and args.generate_johnson_gif:
            logger.info(f"Johnson GIF processing also attempted. Check: {derived_johnson_gif_path}")

    except FileNotFoundError as fnf:
        logger.error(f"File error: {fnf}", exc_info=True)
        sys.exit(1)
    except ValueError as ve:
        logger.error(f"Input or value error: {ve}", exc_info=True)
        sys.exit(1)
    except RuntimeError as rte:
        logger.error(f"Runtime error during processing: {rte}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()