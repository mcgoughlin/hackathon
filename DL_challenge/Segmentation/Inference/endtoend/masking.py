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
from KCD.Detection.Preprocessing.AxialSlices import array_manipulation_utils as amu
#
# nii_label_loc = str(sys.argv[1]) # '/Users/mcgoug01/Downloads/inferences/[4 4 4]mm'
# nii_image_loc = str(sys.argv[2]) # '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/coreg_ncct/images'
# save_loc =  str(sys.argv[3]) # '/Users/mcgoug01/Downloads/masked_coreg_ncct/images'


def mask_dataset(nii_image_loc,nii_label_loc,save_loc):
    label_list = os.listdir(nii_label_loc)

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    #we want to dilate label 40x40x40mm, so 10x10x10 voxels
    for nii_label in label_list:
        # for nii_label in nii_image_list:
        print(nii_label)
        if '.DS_Store' in nii_label:
            continue

        try:
            image = nib.load(os.path.join(nii_image_loc, nii_label))
            label = nib.load(os.path.join(nii_label_loc, nii_label))
        except Exception as e:
            print('Error loading image or label for {}'.format(nii_label))
            print(e)
            continue
        nib.as_closest_canonical(image)
        nib.as_closest_canonical(label)
        aff = image.affine
        header = image.header
        label_data = label.get_fdata()
        image_data = image.get_fdata()
        label_spacing = np.abs(image.header['pixdim'][1:4])
        #compute scale_factor to resize label to 2x2x2mm
        scale_factors = np.array(label_spacing) / np.array([2,2,2])

        #resize voxels to 4x4x4mm and dilate label by 10x10x10 voxels
        label_2 = \
        nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(label_data), dim=0), dim=0),
                                  mode='nearest', scale_factor = tuple(scale_factors)).numpy()[0, 0]
        dilated_label = binary_dilation(label_2, iterations=10)

        #return label to original shape
        dilated_label = \
        nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(torch.Tensor(dilated_label), dim=0), dim=0),
                                    mode='nearest', size=label_data.shape).numpy()[0, 0]

        masked_image = np.where(dilated_label == 0, -500, image_data)

        # create masked nifti image and save
        masked_nifti = nib.Nifti1Image(masked_image,affine= aff, header=header)
        nib.as_closest_canonical(masked_nifti)
        nib.save(masked_nifti, os.path.join(save_loc, nii_label))
