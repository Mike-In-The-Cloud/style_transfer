import torch
import logging
from pathlib import Path

# Assuming TransformerNetwork is in style_transfer.models.johnson_transformer
# Adjust import path if your project structure is different for models
from ..models.johnson_transformer import TransformerNetwork, TransformerNetworkTanh

logger = logging.getLogger(__name__)

def load_johnson_model(weights_path: str, device: torch.device) -> TransformerNetwork:
    """
    Loads a pre-trained Johnson (Fast Neural Style) TransformerNetwork model.

    Args:
        weights_path (str): Path to the .pth file containing the model weights.
        device (torch.device): The device (e.g., 'cuda', 'cpu') to load the model onto.

    Returns:
        TransformerNetwork: The loaded and initialized model, ready for evaluation.

    Raises:
        FileNotFoundError: If the weights_path does not exist.
        RuntimeError: If there's an issue loading the state_dict (e.g., mismatch).
    """
    weights_p = Path(weights_path)
    if not weights_p.exists():
        err_msg = f"Johnson model weights file not found: {weights_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Loading Johnson model weights from: {weights_path} onto device: {device}")

    # Instantiate the network.
    # For now, we assume standard TransformerNetwork.
    # If you have models trained with TransformerNetworkTanh, you might need a way to specify which to load.
    # One simple way is to try loading into TransformerNetwork, and if it fails due to Tanh keys,
    # try TransformerNetworkTanh, or have a naming convention / separate loader.
    # For simplicity, let's assume standard TransformerNetwork first.

    # TODO: Decide if we need to distinguish between TransformerNetwork and TransformerNetworkTanh loading
    # based on weights or a parameter. For now, defaulting to TransformerNetwork.
    model = TransformerNetwork().to(device)

    try:
        # Load the state dictionary
        state_dict = torch.load(weights_p, map_location=device)
        model.load_state_dict(state_dict)
        model.eval() # Set to evaluation mode
        logger.info(f"Successfully loaded Johnson model weights into TransformerNetwork from {weights_path}.")
        return model
    except FileNotFoundError:
        # This is already checked above, but as a safeguard for torch.load itself.
        logger.error(f"Pytorch could not find weights file (should have been caught earlier): {weights_path}")
        raise
    except RuntimeError as e:
        logger.error(f"Error loading state_dict for Johnson model from {weights_path}: {e}")
        logger.error("This might be due to a mismatch between the model architecture and the saved weights,")
        logger.error("or the weights file might be for a TransformerNetworkTanh model.")
        # You could try loading into TransformerNetworkTanh here as a fallback if desired:
        # logger.info("Attempting to load into TransformerNetworkTanh as a fallback...")
        # try:
        #     model_tanh = TransformerNetworkTanh().to(device)
        #     model_tanh.load_state_dict(state_dict)
        #     model_tanh.eval()
        #     logger.info("Successfully loaded into TransformerNetworkTanh as fallback.")
        #     return model_tanh
        # except Exception as e_tanh:
        #     logger.error(f"Fallback to TransformerNetworkTanh also failed: {e_tanh}")
        raise RuntimeError(f"Failed to load Johnson model weights from {weights_path}. Error: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors during loading
        logger.error(f"An unexpected error occurred while loading Johnson model: {e}", exc_info=True)
        raise

# If you also need to load VGG encoder or AdaIN decoder separately (though StyleTransferModel handles this),
# those functions would go here too, e.g.:
# def load_vgg_encoder(weights_path: str, device: torch.device) -> Encoder:
#     # ... implementation ...
# def load_adain_decoder(weights_path: str, device: torch.device) -> Decoder:
#     # ... implementation ...