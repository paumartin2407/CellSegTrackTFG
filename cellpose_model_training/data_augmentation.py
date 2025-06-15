import os
import shutil
import random
import numpy as np
import cv2
import re
from glob import glob
import imgaug.augmenters as iaa

# Fix deprecated numpy alias
np.bool = np.bool_

def natural_keys(text):
    """
    Sorting helper to sort filenames numerically.
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def augment_data_geometric(input_folder, output_folder, geo_aug_probability=0.5):
    """
    Apply geometric augmentations (flip, rotate, scale, perspective) to images and masks.
    """
    augmenters = [
        iaa.SomeOf((1, 3), [
            iaa.Fliplr(1.0),
            iaa.Flipud(1.0),
            iaa.Rot90((1, 3)),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.PerspectiveTransform(scale=(0.01, 0.1)),
        ], random_order=True)
    ]
    seq = iaa.Sequential(augmenters)

    image_files = sorted(glob(os.path.join(input_folder, "frame_*.tif")), key=natural_keys)
    label_files = sorted(glob(os.path.join(input_folder, "frame_*_seg.npy")), key=natural_keys)

    for img_path, lbl_path in zip(image_files, label_files):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        lbl = np.load(lbl_path, allow_pickle=True).item()['masks']

        # Save original images and masks
        img_name = os.path.basename(img_path)
        lbl_name = os.path.basename(lbl_path).replace("_seg.npy", "_masks.tif")
        cv2.imwrite(os.path.join(output_folder, img_name), img)
        cv2.imwrite(os.path.join(output_folder, lbl_name), lbl)

        # Apply geometric augmentation with a certain probability
        if random.random() < geo_aug_probability:
            seq_det = seq.to_deterministic()
            aug_img = seq_det(image=img)
            aug_lbl = seq_det(image=lbl)

            img_aug_name = img_name.replace(".tif", "_geo_aug.tif")
            lbl_aug_name = lbl_name.replace("_masks.tif", "_geo_aug_masks.tif")

            cv2.imwrite(os.path.join(output_folder, img_aug_name), aug_img)
            cv2.imwrite(os.path.join(output_folder, lbl_aug_name), aug_lbl)

def augment_data_photometric(input_folder, output_folder, phot_aug_probability=0.5):
    """
    Apply photometric augmentations (contrast, blur, etc.) to images only.
    Masks are saved unchanged.
    """
    photometric_augmenters = [
        iaa.GammaContrast((0.5, 2.0)),
        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
        iaa.LogContrast(gain=(0.6, 1.4)),
        iaa.LinearContrast((0.4, 1.6)),
        iaa.AverageBlur(k=(2, 11)),
        iaa.MotionBlur(k=5),
        iaa.ElasticTransformation(alpha=(0, 2.5), sigma=(0, 0.12))
    ]

    image_files = sorted(glob(os.path.join(input_folder, "frame_*.tif")), key=natural_keys)
    label_files = sorted(glob(os.path.join(input_folder, "frame_*_seg.npy")), key=natural_keys)

    for img_path, lbl_path in zip(image_files, label_files):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        lbl = np.load(lbl_path, allow_pickle=True).item()['masks']

        # Save original images and masks if not already saved
        img_name = os.path.basename(img_path)
        lbl_name = os.path.basename(lbl_path).replace("_seg.npy", "_masks.tif")
        cv2.imwrite(os.path.join(output_folder, img_name), img)
        cv2.imwrite(os.path.join(output_folder, lbl_name), lbl)

        # Apply photometric augmentation with a certain probability
        if random.random() < phot_aug_probability:
            aug = random.choice(photometric_augmenters)
            aug_img = aug(image=img)

            img_aug_name = img_name.replace(".tif", "_phot_aug.tif")
            lbl_aug_name = lbl_name.replace("_masks.tif", "_phot_aug_masks.tif")

            cv2.imwrite(os.path.join(output_folder, img_aug_name), aug_img)
            cv2.imwrite(os.path.join(output_folder, lbl_aug_name), lbl)

def augment_data(input_base_folder="pretrain", output_base_folder="train"):
    """
    Full pipeline to augment training images and organize data for Cellpose training.
    - Applies geometric and photometric augmentations to training images.
    - Copies testing images without augmentation.
    """

    # Define paths
    original_train_folder = os.path.join(input_base_folder, "train")
    original_test_folder = os.path.join(input_base_folder, "test")
    augmented_train_folder = os.path.join(output_base_folder, "train")
    copied_test_folder = os.path.join(output_base_folder, "test")

    # Clean output folder if exists
    if os.path.exists(output_base_folder):
        shutil.rmtree(output_base_folder)
    os.makedirs(augmented_train_folder, exist_ok=True)
    os.makedirs(copied_test_folder, exist_ok=True)

    # Apply augmentations to training set
    print("Applying augmentations to training data...")
    augment_data_geometric(original_train_folder, augmented_train_folder)
    augment_data_photometric(original_train_folder, augmented_train_folder)

    # Copy test data without augmentations
    print("Copying test data...")
    test_image_files = glob(os.path.join(original_test_folder, "*.tif"))
    test_label_files = glob(os.path.join(original_test_folder, "*.npy"))

    for file_path in test_image_files + test_label_files:
        shutil.copy(file_path, copied_test_folder)

    print("Data augmentation and organization complete!")
