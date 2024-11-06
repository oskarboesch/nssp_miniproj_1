import nibabel as nib
from nilearn.decomposition import CanICA
from nilearn.image import resample_img
import os

def main():
    ica_result_dir = os.path.join("..", "..", "results", "ica")
    # Load the functional image
    data_img = nib.load("../../data/preprocessed/sub-control01/func/preprocessed_data.nii.gz")
    # RUN ICA
    n_components = 10
    ica = CanICA(n_components=n_components)
    ica.fit(data_img)

    # Retrieve the independent components in brain space
    components_img = ica.components_img_
    # Save the components to a Nifti file
    components_img.to_filename(os.path.join(ica_result_dir, "ica_components.nii.gz"))
    # Load the anatomical image
    anat_img = nib.load("../../data/raw/sub-control01/anat/sub-control01_T1w.nii.gz")

    # Resample the components image to match the anatomical image's shape and affine
    mapped_components_img = resample_img(components_img, target_affine=anat_img.affine, target_shape=anat_img.shape)

    # Save the mapped components image to a new NIfTI file
    mapped_components_img.to_filename(os.path.join(ica_result_dir, "ica_components_mapped.nii.gz"))



if __name__ == '__main__':
    main()