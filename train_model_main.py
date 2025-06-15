import sys
import os
from cellpose_model_training.data_augmentation import augment_data
from cellpose_model_training.cellpose_training import train_model

def main(model_name: str):
    """
    Main pipeline to augment data and train a custom Cellpose model.

    Args:
        model_name (str): Name to assign to the trained model (e.g., 'oCyto400x400_aug').
    """

    # ---------------------- Settings ----------------------
    input_base_folder = "cellpose_model_training/pretrain"  # Where original_dataset is located
    output_augmented_folder = "cellpose_model_training/train"

    train_dir = os.path.join(output_augmented_folder, "train")
    test_dir = os.path.join(output_augmented_folder, "test")
    model_save_dir = "cellpose_model"  # Save the model in the same train folder

    # Training hyperparameters
    n_epochs = 400
    nimg_per_epoch = 400
    use_sgd = True
    channels = [0, 0]  # Grayscale images

    # ---------------------- Step 1: Data Augmentation ----------------------
    print("Starting data augmentation...")
    augment_data(input_base_folder=input_base_folder, output_base_folder=output_augmented_folder)
    print("Data augmentation completed successfully!")

    # ---------------------- Step 2: Train the Model -----------------------
    print(f"Starting training with model name '{model_name}'...")
    train_model(
        train_dir=train_dir,
        test_dir=test_dir,
        model_save_dir=model_save_dir,
        model_name=model_name,
        n_epochs=n_epochs,
        nimg_per_epoch=nimg_per_epoch,
        use_sgd=use_sgd,
        channels=channels
    )
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model_main.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
