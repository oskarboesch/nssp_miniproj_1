from nipype.interfaces import fsl
from nilearn import datasets
from nilearn.image import concat_imgs
import os
import glob
import nibabel as nib


def coregister_anat_to_mni(anat_brain_img_path, output_dir, anat_head_img_path):
    """
    Coregisters an anatomical image to MNI space.

    Parameters:
    - anat_img_path: str, path to the anatomical (T1-weighted) image (skull-stripped).
    - output_dir: str, directory to save the coregistered outputs.
    - full_head_img_path: str, path to the full-head anatomical image.

    Returns:
    - anat2mni_mat: str, path to the linear affine matrix file.
    - anat2mni_linear: str, path to the skull-stripped anatomical image in MNI space.
    - head2mni_linear: str, path to the full-head anatomical image in MNI space.
    """
    os.makedirs(output_dir, exist_ok=True)
    template_path = "../data/raw/template/MNI152_T1_2mm_brain.nii.gz"

    # Load MNI template if not exists
    if not os.path.exists(template_path):
        mni = datasets.load_mni152_template()
        mni.to_filename(template_path)

    # Paths for outputs
    anat2mni_linear = os.path.join(output_dir, 'anat2mni_linear_brain.nii.gz')
    anat2mni_mat = os.path.join(output_dir, 'anat2mni_affine_brain.mat')
    head2mni_linear = os.path.join(output_dir, 'anat2mni_linear_head.nii.gz')
    
    # Step 1: Register the skull-stripped image to MNI space
    flirt = fsl.FLIRT(
        in_file=anat_brain_img_path,
        reference=template_path,
        out_file=anat2mni_linear,
        out_matrix_file=anat2mni_mat,
        cost="corratio",
        dof=12
    )
    flirt.run()

    # Step 2: Apply the same transform to the full-head image
    flirt_apply = fsl.FLIRT(
        in_file=anat_head_img_path,
        reference=template_path,
        out_file=head2mni_linear,
        in_matrix_file=anat2mni_mat,
        apply_xfm=True
    )
    flirt_apply.run()
    
    return anat2mni_mat, anat2mni_linear, head2mni_linear

def coregister_fmri_to_anat_epireg(fmri_img_path, anat_img_path, output_dir):
    """
    Coregisters fMRI (EPI) data to an anatomical image using FSL's epireg,
    obtaining the transformation matrix from the first volume and applying it to the others.

    Parameters:
    - fmri_img_path: str, path to the preprocessed fMRI (EPI) image (4D).
    - anat_img_path: str, path to the coregistered anatomical image.
    - output_dir: str, directory to save the coregistered output.
    
    Returns:
    - coregistered_fmri_4d: str, path to the single 4D coregistered fMRI image in anatomical space.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Path for the coregistered fMRI image
    fmri2anat_base = os.path.join(output_dir, fmri_img_path.split('/')[-1].replace('.nii.gz', '_to_anat'))
    
    #Save first volume of the fMRI image
    first_vol = os.path.join(output_dir, fmri_img_path.split('/')[-1].replace('.nii.gz', '_first_vol.nii.gz'))
    fsl.ExtractROI(in_file=fmri_img_path, t_min=0, t_size=1, roi_file=first_vol).run()

    # Coregister the first volume to the anatomical image
    epireg = fsl.EpiReg(
        epi=first_vol,
        t1_head=os.path.join(anat_img_path, 'anat2mni_linear_head.nii.gz'),
        t1_brain=os.path.join(anat_img_path, 'anat2mni_linear_brain.nii.gz'),
        echospacing=0.00201,
        out_base=fmri2anat_base
    )
    epireg.run()
    # Clean all unnecessary files in the shape f'{fmri2anat_base}_fast*'
    os.remove(first_vol)
    for file in glob.glob(f'{fmri2anat_base}_fast*'):
        os.remove(file)

    # Load the transformation matrix from the output
    transform_matrix = fmri2anat_base + '.mat'
    
    # Check if transformation matrix exists
    if not os.path.exists(transform_matrix):
        raise FileNotFoundError(f"Transformation matrix not found: {transform_matrix}")

    # Coregister all volumes of the fMRI image to the anatomical image
    coregistered_fmri_4d_path = os.path.join(output_dir, 'coregistered_fmri_4d.nii.gz')
    
    # Initialize a list to hold the paths of coregistered volumes
    coregistered_volumes = []
    
    # Get the number of volumes in the 4D fMRI image
    num_volumes = nib.load(fmri_img_path).shape[-1]
    
    for vol in range(num_volumes):
        # Define paths for individual volumes
        vol_file = os.path.join(output_dir, f'fmri_vol_{vol}.nii.gz')
        
        # Extract the current volume from the 4D fMRI image
        fsl.ExtractROI(in_file=fmri_img_path, t_min=vol, t_size=1, roi_file=vol_file).run()
        
        # Define the output path for the coregistered volume
        coregistered_vol_file = os.path.join(output_dir, f'coregistered_fmri_vol_{vol}.nii.gz')
        
        # Coregister the current volume to the anatomical image using FLIRT
        flirt = fsl.FLIRT(
            in_file=vol_file,
            reference=os.path.join(anat_img_path, 'anat2mni_linear_brain.nii.gz'),
            out_file=coregistered_vol_file,
            in_matrix_file=transform_matrix,
            apply_xfm=True
        )
        
        try:
            flirt.run()
        except Exception as e:
            print(f"FLIRT failed for volume {vol}: {e}")
        
        # Append the path of the coregistered volume
        coregistered_volumes.append(coregistered_vol_file)
        os.remove(vol_file)  # Optionally remove the extracted volume file

    # Use fsl.Merge to create a 4D image from the coregistered volumes
    # concatenate the volumes using from nilearn.image import mean_img, concat_imgs
    merged_img = concat_imgs(coregistered_volumes)
    # Save the concatenated image to a file
    merged_img.to_filename(coregistered_fmri_4d_path)


    # Clean up individual coregistered volumes
    for coreg_vol in coregistered_volumes:
        os.remove(coreg_vol)

    return coregistered_fmri_4d_path
