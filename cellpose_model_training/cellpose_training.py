import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, train
import pickle

train_dir = "data/train/train"         # Directory containing training images and masks
test_dir = "data/train/test"  

def train_model(train_dir="train/train", test_dir="train/test", model_save_dir="cellpose_model_training", model_name="oCyto400x400",
                n_epochs=400, nimg_per_epoch=400, use_sgd=True, channels=[0, 0]):
    """
    Train a custom Cellpose model using specified training and testing datasets.

    Args:
        train_dir (str): Path to training images and masks.
        test_dir (str): Path to testing images and masks.
        model_save_dir (str): Path where the trained model and losses will be saved.
        model_name (str, optional): Name for the saved model. Default is 'oCyto400x400'.
        n_epochs (int, optional): Number of epochs to train. Default is 400.
        nimg_per_epoch (int, optional): Number of images to sample per epoch. Default is 100.
        use_sgd (bool, optional): Whether to use SGD optimizer (if False, uses Adam). Default is False.
        channels (list, optional): List specifying input channels for Cellpose. Default is [0, 0] (grayscale).
    """

    # ---------------------- GPU Check ----------------------------
    use_GPU = core.use_gpu()
    print(f">>> GPU activated? {'YES' if use_GPU else 'NO'}")

    # ---------------------- Logger Setup -------------------------
    logger = io.logger_setup()

    # ---------------------- Load Model ---------------------------
    model = models.CellposeModel(gpu=use_GPU, model_type='cyto3')  # 'cyto3' pretrained, fine-tuned on your data

    # ---------------------- Load Data ----------------------------
    output = io.load_train_test_data(train_dir, test_dir, mask_filter='_masks')
    train_data, train_labels, _, test_data, test_labels, _ = output

    # ---------------------- Train Model --------------------------
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        channels=channels,
        save_path=model_save_dir,
        n_epochs=n_epochs,
        SGD=use_sgd,
        nimg_per_epoch=nimg_per_epoch,
        model_name=model_name
    )

    print(f"Training completed. Model saved to {model_path}.")

    # ---------------------- Save Losses --------------------------
    losses_path = os.path.join(model_save_dir, f"{model_name}_losses.pkl")
    with open(losses_path, "wb") as f:
        pickle.dump({"train_losses": train_losses, "test_losses": test_losses}, f)

    print(f"Training and testing losses saved to {losses_path}.")

