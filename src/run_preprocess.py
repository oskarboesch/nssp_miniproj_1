import os
import nibabel as nib
from nilearn.image import concat_imgs, smooth_img
from nipype.interfaces.fsl import BET, MCFLIRT, SliceTimer
from preprocess.coregistrate import coregister_anat_to_mni, coregister_fmri_to_anat_epireg

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
    skull_stripped_brain = os.path.join(preprocess_anat_dir,"brain_anat.nii.gz")
    bet = BET(in_file=anat_path, out_file=skull_stripped_brain, mask=True)
    bet.run()

    # Perform Spatial Normalization on the anatomical image
    anat2mni_mat, anat2mni_linear, head2mni_linear = coregister_anat_to_mni(skull_stripped_brain, output_dir=preprocess_anat_dir, anat_head_img_path=anat_path)

    # Standardize and save each functional run
    standardized_func_paths = []
    for func_path in func_paths:
        # Standardise the intensity values of the functional image
        img = nib.load(func_path)
        img_standard = (img.get_fdata() - img.get_fdata().mean()) / img.get_fdata().std()
        img_standard_nii = nib.Nifti1Image(img_standard, img.affine)
        img_standard_nii.to_filename(os.path.join(preprocess_func_dir, os.path.basename(func_path)))
        standardized_func_paths.append(os.path.join(preprocess_func_dir, os.path.basename(func_path)))

    # Slice Timing Correction
    slice_timing_corrected_paths = []
    for func_path in standardized_func_paths:
        slicetimer = SliceTimer(
            in_file=func_path,
            out_file=os.path.join(preprocess_func_dir, "slicetimed_" + os.path.basename(func_path)),
            time_repetition=3,
            slice_direction=1,  # 1 for ascending, 2 for descending
            interleaved=True
        )
        slicetimer.run()
        slice_timing_corrected_paths.append(slicetimer.inputs.out_file)

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

    motion_corrected_img_path = os.path.join(preprocess_func_dir, "motion_corrected.nii.gz")

    coregistered_img = coregister_fmri_to_anat_epireg(motion_corrected_img_path,preprocess_anat_dir, output_dir=preprocess_func_dir)
    # Apply smoothing
    coregistered_img = os.path.join(preprocess_func_dir, "coregistered_data.nii.gz")
    smoothed_img = smooth_img(coregistered_img, fwhm=5)

    # Save final output
    preproccessed_img_path = os.path.join(preprocess_func_dir, "preprocessed_data.nii.gz")
    smoothed_img.to_filename(preproccessed_img_path)
    



if __name__ == "__main__":
    preprocess_data()
