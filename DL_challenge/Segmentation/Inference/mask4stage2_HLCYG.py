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

dose = float(sys.argv[1])

os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data/"

# nii_label_loc = str(sys.argv[1]) # '/Users/mcgoug01/Downloads/inferences/[4 4 4]mm'
nii_label_loc = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions_nii/coreg_v3_noised/_[4 4 4]mm'
# nii_image_loc = str(sys.argv[2]) # '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/coreg_ncct/images'
nii_image_loc = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/coreg_v3_HLCYG_{}/images'.format(dose)
# save_loc =  str(sys.argv[3]) # '/Users/mcgoug01/Downloads/masked_coreg_ncct/images'
save_loc = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/masked_coreg_v3_HLCYG_{}/images'.format(dose)
npy_label_list = os.listdir(nii_label_loc)
nii_image_list = os.listdir(nii_image_loc)

if not os.path.exists(save_loc):
    os.makedirs(save_loc)

masking.mask_dataset(nii_image_loc, nii_label_loc, save_loc)

raw_labels_loc = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/coreg_v3_HLCYG_{}/labels'.format(dose)
masked_labels_loc = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/masked_coreg_v3_HLCYG_{}/labels'.format(dose)

if not os.path.exists(masked_labels_loc):
    os.makedirs(masked_labels_loc)

#copy all files from raw_labels_loc to masked_labels_loc
for file in os.listdir(raw_labels_loc):
    os.system('cp {} {}'.format(os.path.join(raw_labels_loc, file), os.path.join(masked_labels_loc, file)))

