#copy nii files from src, send to dest with new filename according to key stored in dataframe csv

import numpy as np
import nibabel as nib
import os
import pandas as pd

csv_path = '/Users/mcgoug01/Downloads/test_set_v2/internaltest_metadata.xlsx'
nii_path = '/Users/mcgoug01/Downloads/test_set_v2/preprocessed_nii'
dest_path = '/Users/mcgoug01/Downloads/test_set_v2/renamed_nii'
affine = np.eye(4)
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

df = pd.read_excel(csv_path)
alternate_files = ['13']
#match filename in nii_path to 'OriginalID' in df, rename nii file to 'StudyID' in df
for nii_file in os.listdir(nii_path)[::-1]:
    if nii_file.startswith('TCGA'):continue
    if nii_file.endswith('.nii.gz'):
        print(nii_file)
        print(df.loc[df['OriginalID'] == nii_file,'StudyID'].values[0])
        print()
        # if originalID is numeric.nii.gz, or begins with 'KiTS', flip in second dimension
        if nii_file.startswith('60.'):
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()
        elif nii_file.startswith('191.'):
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()
        elif nii_file.startswith('772.'):
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()
        elif nii_file.startswith('398.'):
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()[:,:,50:]
        elif nii_file.startswith('398.'):
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()[:,:,100:]
        elif nii_file.startswith('189.'):
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()[:,:,100:]
        elif nii_file.startswith('396.'):
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()[:,:,100:-50]
        elif nii_file.startswith('397.'):
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()
        elif nii_file.startswith('767.'):
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()
        elif nii_file[:-7].isnumeric() or nii_file.startswith('KiTS'):
            if nii_file=='132.nii.gz':
                nii = np.flip(nib.load(os.path.join(nii_path, nii_file)).get_fdata(),1)
                # crop 50 of each axial side and 200 off the top axial
                nii = nii[50:-50,50:-50,250:-300]
                nii = nib.Nifti1Image(nii, affine, dtype=np.float32)
                nib.save(nii,os.path.join(dest_path, df.loc[df['OriginalID'] == nii_file, 'StudyID'].values[0] + '.nii.gz'))
                continue
            elif nii_file=='20.nii.gz':
                nii = np.flip(nib.load(os.path.join(nii_path, nii_file)).get_fdata(),1)
                # crop 50 of each axial side and 200 off the top axial
                nii = nii[50:-50,50:-50,100:-100]
                nii = nib.Nifti1Image(nii, affine, dtype=np.float32)
                nib.save(nii,os.path.join(dest_path, df.loc[df['OriginalID'] == nii_file, 'StudyID'].values[0] + '.nii.gz'))
                continue
            elif nii_file=='133.nii.gz':
                nii = np.flip(nib.load(os.path.join(nii_path, nii_file)).get_fdata(),1)
                # crop 50 of each axial side and 200 off the top axial
                nii = nii[50:-50,50:-50,:-300]
                nii = nib.Nifti1Image(nii, affine, dtype=np.float32)
                nib.save(nii,os.path.join(dest_path, df.loc[df['OriginalID'] == nii_file, 'StudyID'].values[0] + '.nii.gz'))
                continue
            elif nii_file=='131.nii.gz':
                nii = np.flip(nib.load(os.path.join(nii_path, nii_file)).get_fdata(),1)
                # crop 50 of each axial side and 200 off the top axial
                nii = nii[50:-50,50:-50,50:-200]
                nii = nib.Nifti1Image(nii, affine, dtype=np.float32)
                nib.save(nii,os.path.join(dest_path, df.loc[df['OriginalID'] == nii_file, 'StudyID'].values[0] + '.nii.gz'))
                continue
            elif nii_file[:-8] in alternate_files:
                print('alternate file found')
                nii = np.flip(nib.load(os.path.join(nii_path, nii_file)).get_fdata(),1)
                # crop 50 of each axial side and 200 off the top axial
                nii = nii[50:-50,50:-50,100:-100]
                nii = nib.Nifti1Image(nii, affine, dtype=np.float32)
                nib.save(nii,os.path.join(dest_path, df.loc[df['OriginalID'] == nii_file, 'StudyID'].values[0] + '.nii.gz'))
                continue
            nii = np.flip(nib.load(os.path.join(nii_path, nii_file)).get_fdata(),1)
            if nii_file[:-7].isnumeric():
                # crop 50 of each axial side and 200 off the top axial
                nii = nii[50:-50,50:-50,:-200]
            else:
                # crop 50 of each axial side and 100 off the top
                nii = nii[:,:,100:400]

            nii = nib.Nifti1Image(nii, affine, dtype=np.float32)
            nib.save(nii,os.path.join(dest_path, df.loc[df['OriginalID'] == nii_file, 'StudyID'].values[0] + '.nii.gz'))
            continue
        else:
            nii = nib.load(os.path.join(nii_path, nii_file)).get_fdata()

        nii = nib.Nifti1Image(nii, affine,dtype=np.float32)
        nib.save(nii, os.path.join(dest_path, df.loc[df['OriginalID'] == nii_file,'StudyID'].values[0] + '.nii.gz'))