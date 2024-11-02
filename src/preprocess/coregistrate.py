from nipype.interfaces import fsl
from nilearn import datasets
import os

def coregister_fmri_to_mni(fmri_img_path, output_dir):
    """
    Coregister fMRI data to an MNI template using FSL's flirt.

    Parameters:
    - fmri_img_path: str, path to the preprocessed fMRI image.
    - output_dir: str, directory to save the coregistered output.
    
    Returns:
    - coregistered_img: str, path to the coregistered image.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    template_path = "../data/raw/template/MNI152_T1_2mm_brain.nii.gz"

    # If no MNI load template to folder raw/template
    if not os.path.exists(template_path):
        mni = datasets.load_mni152_template()
        mni.to_filename(template_path)


    # Create a FSL FLIRT node for coregistration
    flirt = fsl.FLIRT()
    flirt.inputs.in_file = fmri_img_path
    flirt.inputs.reference = template_path
    flirt.inputs.out_file = os.path.join(output_dir, 'coregistered_fmri_to_mni.nii.gz')
    flirt.inputs.cost = 'corratio'  # Correlation ratio as the cost function
    flirt.inputs.dof = 6  # Degrees of freedom for rigid body transformation

    # Run the coregistration
    flirt.run()

    return flirt.inputs.out_file