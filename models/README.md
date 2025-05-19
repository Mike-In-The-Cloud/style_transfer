# Pre-trained Model Weights

This directory (`models/`) is intended to store the pre-trained model weights required by the Neural Style Transfer application. These model files are **not** included directly in this repository due to their size and licensing but are essential for the application to function.

Please download the necessary files as described below and place them directly into this `models/` directory.

## Required Models

### 1. AdaIN (Adaptive Instance Normalisation) Models

The AdaIN method requires two model files:

* **VGG19 Encoder Weights:** `vgg_normalised.pth`
* **AdaIN Decoder Weights:** `decoder.pth`

**Source & Credits:**
These model weights are typically sourced from the work of Huang and Belongie and implementations such as the one found in the **[naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)** repository. We gratefully acknowledge their contribution. Please refer to their repository for downloading these specific `.pth` files.

-   **`vgg_normalised.pth`**: Contains the weights for the VGG19 encoder, normalised as per the AdaIN paper.
-   **`decoder.pth`**: Contains the weights for the AdaIN decoder network.

### 2. Johnson's Fast Neural Style Transfer Models

Johnson's method (Fast Neural Style Transfer) uses separate pre-trained Transformer Network models for each distinct artistic style. You will need to download `.pth` files corresponding to the styles you wish to use.

**Examples:**

* `candy.pth`
* `mosaic.pth`
* `rain_princess.pth`
* `udnie.pth`
* `tokyo_ghoul_aggressive.pth`
* (and many others)

**Source & Credits:**
These models are based on the work of Johnson et al. Many pre-trained `.pth` files can be found in repositories that implement Fast Neural Style Transfer. A common source is the **[rrmina/fast-neural-style-pytorch](https://github.com/rrmina/fast-neural-style-pytorch)** repository (or forks thereof) and other similar collections available on GitHub. We acknowledge the original authors and the community for providing these pre-trained models.

**Note:** The GUI component of this application will attempt to automatically list any `.pth` files (excluding `vgg_normalised.pth` and `decoder.pth`) found in this `models/` directory for use with the Johnson method.

## Summary

To use the application, ensure this `models/` directory is populated with:

1. `vgg_normalised.pth`
2. `decoder.pth`
3. One or more Johnson style `.pth` files (e.g., `candy.pth`).

Without these files, the application will not be able to perform style transfer.