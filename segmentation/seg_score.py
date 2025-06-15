import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt


#predicted_masks = io.imread('data/Fluo-N2DL-HeLa/01_ST/SEG/man_seg013.tif')

def seg_score(ground_truth_masks, predicted_masks):
    seg_scores = []
    
    # Loop over each ground truth object (assume each object has a unique integer label)
    for label in np.unique(ground_truth_masks):
        if label == 0:
            continue  # Skip background
        
        # Get the reference object (R)
        reference_object = (ground_truth_masks == label)
        
        # Get the predicted object that overlaps the most with the reference object (S)
        matching_pred_object = None
        max_overlap = 0
        
        for pred_label in np.unique(predicted_masks):
            if pred_label == 0:
                continue  # Skip background
            
            # Get the predicted object (S)
            predicted_object = (predicted_masks == pred_label)
            
            # Calculate overlap
            intersection = np.sum(reference_object & predicted_object)
            union = np.sum(reference_object | predicted_object)
            
            # Ensure matching condition is met
            if intersection > 0.5 * np.sum(reference_object):
                # Jaccard Index (IoU)
                jaccard_index = intersection / union
                if jaccard_index > max_overlap:
                    max_overlap = jaccard_index
        
        # If a matching segmented object is found, use its Jaccard score
        if max_overlap > 0:
            seg_scores.append(max_overlap)
    
    # Return the mean SEG score
    if len(seg_scores) > 0:
        return np.mean(seg_scores)
    else:
        return 0.0
    
test = [1, 12, 13, 14, 27, 30, 46, 61, 66, 69, 78, 80, 93]

def compute_seg_score(test, mask_folder, gt_segmentation_folder):
    # Example usage
    # Initialize list to store SEG scores
    seg_scores = []

    # Loop over the files in the SEG folder
    for i in test:
        
        # Load the corresponding predicted mask from cellPose_masks/mask_J.npy
        mask_path = os.path.join(mask_folder, f'frame_{i}_masks.tif')
        predicted_masks = io.imread(mask_path)
        
        # Load the ground truth segmentation (man_segJ.tif)
        gt_path = os.path.join(gt_segmentation_folder, f'frame_{i}_masks.tif')
        ground_truth_masks = io.imread(gt_path)
        
        # Calculate SEG for this image
        seg_score = seg_score(ground_truth_masks, predicted_masks)
        seg_scores.append(seg_score)
        
        #print(f"Calculated SEG score for image index {i}: {seg_score}")

    return(seg_scores)


gt_segmentation_folder = 'seg_gt'
mask_folder_cyto3 = 'seg_cyto3'
mask_folder_cyto_trained = 'seg_cyto400x400'

seg_score_cyto3 = compute_seg_score(test, mask_folder_cyto3, gt_segmentation_folder)
seg_score_cyto400x400 = compute_seg_score(test, mask_folder_cyto_trained, gt_segmentation_folder)

def plot_scores(test, score1, score2):
    # create a sequence 0,1,2,... of the same length as test
    x = list(range(len(test)))

    plt.figure()
    # plot using x as positions, but marker at each point
    plt.plot(x, score1, marker='o', linestyle='-', label='Score cyto3')
    plt.plot(x, score2, marker='o', linestyle='-', label='Score trained model')
    
    # now replace the tick labels with your actual test integers
    plt.xticks(x, test)
    
    plt.xlabel('Test Index')
    plt.ylabel('Score')
    plt.title('Test Scores')
    plt.legend()
    plt.show()


plot_scores(test, seg_score_cyto3, seg_score_cyto400x400)

