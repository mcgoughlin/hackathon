# quick script that loads an .nii.gz image and its corresponding seg label, dilates the label,
# and then masks the image with the dilated label.

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
import torch
import torch.nn as nn
from scipy import stats
import sys
from KCD.Segmentation.Inference.endtoend import masking

os.environ['OV_DATA_BASE'] = "/home/wcm23/rds/hpc-work/FineTuningKITS23"

# nii_label_loc = str(sys.argv[1]) # '/Users/mcgoug01/Downloads/inferences/[4 4 4]mm'
nii_label_loc = '/home/wcm23/rds/hpc-work/FineTuningKITS23/predictions_nii/coreg_v4_noised/_[4 4 4]mm'
# nii_image_loc = str(sys.argv[2]) # '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/coreg_ncct/images'
nii_image_loc = '/home/wcm23/rds/hpc-work/FineTuningKITS23/raw_data/coreg_v4_noised/images'
# save_loc =  str(sys.argv[3]) # '/Users/mcgoug01/Downloads/masked_coreg_ncct/images'
save_loc = '/home/wcm23/rds/hpc-work/FineTuningKITS23/raw_data/masked_coreg_v4_noised/images'
npy_label_list = os.listdir(nii_label_loc)
nii_image_list = os.listdir(nii_image_loc)

if not os.path.exists(save_loc):
    os.makedirs(save_loc)

masking.mask_dataset(nii_image_loc, nii_label_loc, save_loc)

nib.save(masked_nifti, os.path.join(save_loc, nii_label))
