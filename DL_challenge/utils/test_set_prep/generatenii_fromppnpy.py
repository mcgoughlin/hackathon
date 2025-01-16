# convert numpy files of 3D array into CT scans with an affine matrix of 1x1x1 using nibabel

import numpy as np
import nibabel as nib
import os

npy_path = '/Users/mcgoug01/Downloads/test_set_v2/preprocessed_npy'
nii_path = '/Users/mcgoug01/Downloads/test_set_v2/preprocessed_nii'

if not os.path.exists(nii_path):
    os.makedirs(nii_path)

for npy_file in os.listdir(npy_path):
    if npy_file.endswith('.npy'):
        npy = np.flip(np.flip(np.rot90(np.load(os.path.join(npy_path, npy_file)).astype(np.float32)[0].swapaxes(0,2),3),0),1)
        #count number of voxels in npy
        nv = np.prod(npy.shape)
        if nv > 700**3:
            # use shape to find the a 600x600x600 cube in each dimension
            # cube should be centered 200 pixels above the axial centre
            c = np.array(npy.shape)//2
            npy = npy[c[0]-300:c[0]+300,c[1]-300:c[1]+300,c[2]-100:c[2]+500]
        print(npy_file)
        affine = np.eye(4)
        nii = nib.Nifti1Image(npy, affine,dtype=np.float32)
        print()
        nib.save(nii, os.path.join(nii_path, npy_file.replace('.npy', '.nii.gz')))