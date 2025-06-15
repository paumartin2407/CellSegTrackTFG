import os
import warnings, re

warnings.filterwarnings(
    "ignore",
    message=re.escape("You are using `torch.load` with `weights_only=False`"),
    category=FutureWarning,
    module=r"torch\.serialization"
)

from cellpose import models, core, plot
from skimage.io import imread, imsave

def compute_masks(video_name, model_path=None):
    """
    Applies a trained Cellpose model to generate masks from preprocessed frames.
    
    Args:
        video_name (str): Name of the folder containing preprocessed frames (e.g., 'exp9').
        processed_path (str): Base path where preprocessed frames are stored.
        model_path (str): Path to a custom Cellpose trained model. If None, uses built-in model.
    """

    # Define input and output paths
    input_dir = os.path.join('data/processed', video_name, "green_channel")
    masks_dir = os.path.join('data/masks', f"{video_name}")
    
    # Ensure output directory exists
    os.makedirs(masks_dir, exist_ok=True)

    # Automatically detect if GPU is available
    use_gpu = core.use_gpu()
    print(f"GPU activated? {'YES' if use_gpu else 'NO'}")

    # Initialize the Cellpose model
    model = models.CellposeModel(
        model_type=None, 
        gpu=use_gpu, 
        pretrained_model=model_path
    )

    # List all frames inside the input directory
    frames = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.tif')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    j = 0

    for frame_filename in frames:
        img_path = os.path.join(input_dir, frame_filename)

        # Load image
        img = imread(img_path)

        # Apply Cellpose model
        masks = model.eval(img, channels=[0, 0], diameter=23.0)

        # Save only the masks as .tif
        output_mask_path = os.path.join(masks_dir, f"{os.path.splitext(frame_filename)[0]}_masks.tif")
        imsave(output_mask_path, masks[0], check_contrast=False)

        j+=1

    print(f"Saved mask {j} files for {video_name}.")


