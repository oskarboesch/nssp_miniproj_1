from nilearn.image import load_img
from nilearn.datasets import fetch_atlas_aal
from nilearn.image import coord_transform
from nilearn.maskers import NiftiLabelsMasker
import numpy as np
import nibabel as nib

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