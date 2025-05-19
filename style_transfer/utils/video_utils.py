import imageio
import os
import glob
import logging
import torch
from pathlib import Path
from torchvision import transforms

logger = logging.getLogger(__name__)

def create_gif_from_frames(frames_dir: str, output_gif_path: str, frame_duration: float = 0.1, loop: int = 0, ping_pong: bool = False):
    """
    Creates a GIF from a sequence of image frames.

    Args:
        frames_dir: Directory containing the image frames (e.g., PNGs).
        output_gif_path: Path to save the generated GIF.
        frame_duration: Duration of each frame in seconds (e.g., 0.1 for 10 FPS).
        loop: Number of times the GIF should loop (0 for infinite).
        ping_pong: If True, the frame sequence will play forwards then backwards (excluding duplicate end/start frames).
    """
    images = []
    # Common image file extensions
    frame_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    frame_files = []
    for pattern in frame_patterns:
        frame_files.extend(glob.glob(os.path.join(frames_dir, pattern)))

    frame_files.sort() # Sort alphabetically/numerically to ensure correct order

    if not frame_files:
        logger.warning(f"No frames found in {frames_dir} with patterns {frame_patterns}")
        return False

    if ping_pong and len(frame_files) >= 2:
        # Create the reversed sequence, excluding the last frame of the original
        # and the first frame of the reversed part (which is the same as the original's last)
        # Example: [f0, f1, f2, f3] -> [f0, f1, f2, f3, f2, f1]
        # reversed_part = frame_files[-2:0:-1] # This slice is [frame_at_len-2, frame_at_len-3, ..., frame_at_1]
        # To get [f3, f2, f1, f0] -> reversed_frames = frame_files[::-1]
        # We want frames_files + reversed_frames[1:-1] if we want to exclude start/end of reverse.
        # Simpler: forward part is frame_files. Backward part is frame_files reversed, excluding its first and last element.
        # If frame_files = [a,b,c,d], reversed = [d,c,b,a]. We want [a,b,c,d] + [c,b]
        # So, it's frame_files + frame_files[-2:0:-1]
        reversed_part = frame_files[-2:0:-1] # This slice is [frame_at_len-2, frame_at_len-3, ..., frame_at_1]
        logger.info(f"Ping-pong mode enabled. Adding {len(reversed_part)} frames for reverse sequence.")
        frame_files.extend(reversed_part)
    elif ping_pong and len(frame_files) < 2:
        logger.warning("Ping-pong effect requires at least 2 frames. Proceeding without ping-pong.")

    logger.info(f"Found {len(frame_files)} frames (after potential ping-pong) in {frames_dir}. Creating GIF: {output_gif_path}")
    try:
        for filename in frame_files:
            images.append(imageio.imread(filename))

        # Ensure output directory exists for GIF
        output_dir = os.path.dirname(output_gif_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        imageio.mimsave(output_gif_path, images, duration=frame_duration, loop=loop)
        logger.info(f"GIF successfully saved to {output_gif_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create GIF at {output_gif_path}: {e}", exc_info=True)
        return False

def generate_interpolated_frames(
    content_tensor_orig: torch.Tensor,
    stylized_tensor_orig: torch.Tensor,
    num_frames: int,
    output_dir_path: str,
    gif_style_intensity: float = 1.0
):
    """
    Generates and saves a sequence of frames by linearly interpolating
    between a content tensor and a stylized tensor, with adjustable style intensity for the GIF endpoint.

    Args:
        content_tensor_orig: The original content image tensor (e.g., from preprocess_image, range [0,255]).
        stylized_tensor_orig: The raw stylized image tensor from the Johnson model (range approx [0,255]).
        num_frames: The number of intermediate frames to generate.
        output_dir_path: Directory to save the generated frame images.
        gif_style_intensity: Controls how much of the style is applied at the end of the GIF (0.0 to 1.0).
                             1.0 means full style, 0.0 means final frame is content.
    """
    if num_frames <= 0:
        logger.warning("num_frames must be positive. Skipping frame generation.")
        return

    output_path = Path(output_dir_path)
    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Generating {num_frames} interpolated frames in {output_dir_path}...")

    # Squeeze batch dim and move to CPU
    content_t = content_tensor_orig.cpu().squeeze(0)
    stylized_t = stylized_tensor_orig.cpu().squeeze(0)

    # Clamp inputs to [0, 255] and normalize to [0, 1] for interpolation
    content_t_norm = torch.clamp(content_t, 0, 255) / 255.0
    stylized_t_norm = torch.clamp(stylized_t, 0, 255) / 255.0

    # Calculate the effective target for the stylized end of the GIF
    # based on gif_style_intensity
    effective_stylized_target_norm = (1.0 - gif_style_intensity) * content_t_norm + \
                                     gif_style_intensity * stylized_t_norm
    effective_stylized_target_norm = torch.clamp(effective_stylized_target_norm, 0, 1)

    # Optional: A very gentle damping for the stylized end of the GIF if needed later.
    # For now, let's try without it, relying on the user to pick a good model.
    # stylized_t_norm = stylized_t_norm * 0.98 # Example damping
    # stylized_t_norm = torch.clamp(stylized_t_norm, 0, 1)

    to_pil = transforms.ToPILImage()

    for i in range(num_frames):
        if num_frames == 1:
            t = 1.0
        else:
            t = i / (num_frames - 1)

        interpolated_tensor = (1 - t) * content_t_norm + t * effective_stylized_target_norm
        interpolated_tensor = torch.clamp(interpolated_tensor, 0, 1) # Ensure stays in [0,1]

        frame_filename = output_path / f"frame_{i:04d}.png"
        try:
            pil_frame = to_pil(interpolated_tensor) # Input is [0,1] float tensor
            pil_frame.save(frame_filename)
        except Exception as e:
            logger.error(f"Failed to save interpolated frame {frame_filename}: {e}")
    logger.info(f"Finished generating interpolated frames.")

if __name__ == '__main__':
    # This is a simple test block.
    # To test:
    # 1. Create a directory (e.g., 'test_frames').
    # 2. Put a few numbered PNG or JPG images inside it (e.g., frame_001.png, frame_002.png).
    # 3. Run: python -m style_transfer.utils.video_utils

    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    test_frames_input_dir = "test_frames_temp_delete_me"
    test_gif_output_path = "test_sequence_delete_me.gif"

    if not os.path.exists(test_frames_input_dir):
        os.makedirs(test_frames_input_dir)
        logger.info(f"Created {test_frames_input_dir} for testing.")
        logger.info(f"Please add some .png or .jpg images (e.g., frame_00.png, frame_01.png) to this directory.")
        # Example: create dummy files (requires Pillow)
        try:
            from PIL import Image, ImageDraw
            for i in range(5):
                img = Image.new('RGB', (100, 100), color = ('red' if i % 2 == 0 else 'blue'))
                draw = ImageDraw.Draw(img)
                draw.text((10,10), f"Frame {i}", fill=(255,255,0))
                img.save(os.path.join(test_frames_input_dir, f"frame_{i:02d}.png"))
            logger.info(f"Created dummy frames in {test_frames_input_dir}")

            if create_gif_from_frames(test_frames_input_dir, test_gif_output_path, frame_duration=0.5):
                logger.info(f"Test GIF created: {test_gif_output_path}")
            else:
                logger.error(f"Test GIF creation failed.")

        except ImportError:
            logger.warning("Pillow not installed. Cannot create dummy frames for testing video_utils automatically.")
            logger.info("To test, manually create some image files in a directory and call create_gif_from_frames().")
        except Exception as e:
            logger.error(f"Error during test setup: {e}")
    else:
        logger.info(f"Test directory {test_frames_input_dir} already exists. Assuming it has frames.")
        if create_gif_from_frames(test_frames_input_dir, test_gif_output_path, frame_duration=0.5):
            logger.info(f"Test GIF created: {test_gif_output_path}")
        else:
            logger.error(f"Test GIF creation failed.")
    # To clean up, you would manually delete test_frames_temp_delete_me and test_sequence_delete_me.gif