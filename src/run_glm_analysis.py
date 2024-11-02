from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_stat_map, plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import mean_img, load_img
from nilearn.datasets import fetch_atlas_aal
from nilearn.image import coord_transform
from nilearn.maskers import NiftiLabelsMasker
import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def find_brain_regions(thresholded_map: nib.Nifti1Image) -> None:
    """
    Find the brain region in the AAL atlas with the maximum z-score in the thresholded map.
    """
    z_data = thresholded_map.get_fdata()

    # Find the coordinates of the voxel with the maximum z-score
    max_voxel_coords = np.unravel_index(np.argmax(z_data), z_data.shape)
    max_z_value = np.max(z_data)
    print(f"Max z-score: {max_z_value} at voxel coordinates: {max_voxel_coords}")

    aal_atlas = fetch_atlas_aal()
    aal_img = load_img(aal_atlas.maps)
    aal_labels = aal_atlas.labels

    affine = thresholded_map.affine
    mni_coords = coord_transform(*max_voxel_coords, affine)

    # Create a masker for the AAL atlas
    masker = NiftiLabelsMasker(aal_img, standardize=False)
    aal_data = masker.fit_transform(thresholded_map)

    # Get the region label
    region_index = np.argmax(aal_data)
    region_label = aal_labels[region_index]

    print(f"Maximal contrast region: {region_label} at coordinates {mni_coords}")

def main() -> None:
    # Define paths
    preprocess_dir = os.path.join("data", "preprocessed", "sub-control01")
    figures_dir = os.path.join("results", "figures")

    # Load the design matrix
    design_matrix = pd.read_csv(os.path.join(preprocess_dir, "design_matrix.csv"))

    # Load the preprocessed functional image
    img = nib.load(os.path.join(preprocess_dir, "preprocessed_data.nii.gz"))
    TR = img.header.get_zooms()[3].item()

    # Perform the GLM analysis
    fmri_glm = FirstLevelModel(t_r=TR, hrf_model='spm + derivative', standardize=False)
    fmri_glm = fmri_glm.fit(img, design_matrices=design_matrix)

    # Plot and save the design matrix
    plt.figure(figsize=(12, 6))
    plot_design_matrix(design_matrix)
    plt.savefig(os.path.join(figures_dir, "design_matrix_glm_analysis.png"))

    # Compute the contrast
    _mean_img = mean_img(img)
    contrast = np.eye(design_matrix.shape[1])[0] - np.eye(design_matrix.shape[1])[1]
    z_map = fmri_glm.compute_contrast(contrast, output_type='z_score')

    # Plot and save the Z-map
    plot_stat_map(z_map, bg_img=_mean_img, threshold=2, display_mode='z', cut_coords=[-30, 5, 14], black_bg=True, title='Active minus Rest (Z>2)')
    plt.savefig(os.path.join(figures_dir, "z_map.png"))

    # Apply FPR correction and save the thresholded map
    cluster_size = 10
    thresholded_map, threshold = threshold_stats_img(z_map, alpha=0.05, height_control='fpr', cluster_threshold=cluster_size)
    thresholded_map.to_filename(os.path.join(preprocess_dir, "thresholded_map.nii.gz"))
    _mean_img.to_filename(os.path.join(preprocess_dir, "mean_img.nii.gz"))
    plot_stat_map(thresholded_map, bg_img=_mean_img, threshold=threshold, title="Thresholded Z-map (Z>{threshold})", display_mode='z', cut_coords=[-30, 5, 14], colorbar=True)
    plt.savefig(os.path.join(figures_dir, "fpr_z_map.png"))
    plt.close()

    # Find the brain region with the maximum z-score in the thresholded map
    find_brain_regions(thresholded_map)

if __name__ == "__main__":
    main()