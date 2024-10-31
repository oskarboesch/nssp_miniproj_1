import os
import nibabel as nib
from nilearn.image import concat_imgs, smooth_img
from nipype.interfaces.fsl import BET, MCFLIRT

def preprocess_data():
   # Define paths relative to the src folder
    base_dir = os.path.join("..", "data", "raw", "sub-control01")  # Navigate up one directory and then into data
    preprocess_dir = os.path.join("..", "data", "preprocessed", "sub-control01")  # Preprocess folder
    preprocess_anat_dir = os.path.join(preprocess_dir, "anat")  # Anatomical folder
    preprocess_func_dir = os.path.join(preprocess_dir, "func")  # Functional folder
    anat_path = os.path.join(base_dir, "anat", "sub-control01_T1w.nii.gz")
    func_paths = [
        os.path.join(base_dir, "func", "sub-control01_task-music_run-1_bold.nii.gz"),
        os.path.join(base_dir, "func", "sub-control01_task-music_run-2_bold.nii.gz"),
        os.path.join(base_dir, "func", "sub-control01_task-music_run-3_bold.nii.gz")
    ]
    # Create preprocess directory if it doesn't exist
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)
    if not os.path.exists(preprocess_anat_dir):
        os.makedirs(preprocess_anat_dir)
    if not os.path.exists(preprocess_func_dir):
        os.makedirs(preprocess_func_dir)

    
    # Perform skull stripping
    bet = BET(in_file=anat_path, out_file=os.path.join(preprocess_anat_dir,"brain_anat.nii.gz"), mask=True)
    bet.run()

    # Perform standardisation for each run
    for func_path in func_paths:
        # Standardise the intensity values of the functional image
        img = nib.load(func_path)
        img_standard = (img.get_fdata() - img.get_fdata().mean()) / img.get_fdata().std()
        img_standard_nii = nib.Nifti1Image(img_standard, img.affine)
        img_standard_nii.to_filename(os.path.join(preprocess_func_dir, os.path.basename(func_path)))

    # Load functional images
    func_imgs = [nib.load(func_path) for func_path in func_paths]
    
    # Concatenate functional runs
    concat_img = concat_imgs(func_imgs)

    # Save the concatenated image to a file
    concat_img_path = os.path.join(preprocess_func_dir, "concatenated_func.nii.gz")
    concat_img.to_filename(concat_img_path)

    # Motion correction
    mcflirt = MCFLIRT(in_file=concat_img_path, out_file=os.path.join(preprocess_func_dir, "motion_corrected.nii.gz"))
    mcflirt.run()

    # Apply smoothing
    smoothed_img = smooth_img(os.path.join(preprocess_func_dir, "motion_corrected.nii.gz"), fwhm=5)

    # Save final output
    smoothed_img.to_filename(os.path.join(preprocess_func_dir, "preprocessed_data.nii.gz"))


if __name__ == "__main__":
    preprocess_data()
