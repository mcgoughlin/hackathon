# extract list of .nii.gz files in homepath
# then, loop through each of them and resize using torch.nn.functional.interpolate to 1mm isotropic
# then, convert to numpy array, apply scipy.ndimage labels, and find the largest region
# then, if the size of the largest region is less than 4189, save the image in small_pred_path

import os
import torch
import numpy as np
import scipy.ndimage
import nibabel as nib
import shutil
import pandas as pd

checkpath = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/error_analysis/high_acc/reviewed_potential_misses.csv'
checkhomepath = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/error_analysis/high_acc/potential_misses_postprocessed'
homepath = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/error_analysis/high_acc/post_processed_inferences'
small_pred_path = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/error_analysis/high_acc/potential_misses_postprocessed_hc'
sigfindings_path = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/sig_findings.csv'
insigfindings_path = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/insig_findings.csv'
cont_pred_path = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/error_analysis/high_acc/all_preds_cont'


#open both sigfindings and insigfindings
sigfindings = pd.read_csv(sigfindings_path)
insigfindings = pd.read_csv(insigfindings_path)
sigfindings['IDUniqueStudyID'] = sigfindings['IDUniqueStudyID'].apply(lambda x: int(x))
insigfindings['IDUniqueStudyID'] = insigfindings['IDUniqueStudyID'].apply(lambda x: int(x))
checkfiles = pd.read_csv(checkpath)['case (>0.95)'].values.astype(int)
checkfiles2 = os.listdir(checkhomepath)

if not os.path.exists(small_pred_path):
    os.makedirs(small_pred_path)

for file in os.listdir(homepath):
    if file.endswith('.nii.gz'):
        #extract first 8 characters of filename
        ykst_code = int(file[:8])
        # if ykst_code in IDUniqueStudyID column of sigfindings or insigfindings, continue
        # if ykst_code in sigfindings['IDUniqueStudyID'].values or ykst_code in insigfindings['IDUniqueStudyID'].values:
        #     print(f'{file} is a known finding')
        #     continue

        if ykst_code in sigfindings['IDUniqueStudyID'].values:
            print(f'{file} is a known finding')
            continue

        if file in checkfiles2:
            print(f'{file} is in checkhomepath')
            continue

        #if file in checkpath, continue
        if ykst_code in checkfiles:
            print(f'{file} is in checkpath')
            continue

        # if ykst_code in sigfindings['IDUniqueStudyID'].values:
        #     print(f'{file} is a known finding')
        #     continue
        img = nib.load(os.path.join(homepath, file))
        cont_pred = nib.load(os.path.join(cont_pred_path, file))
        unscaled_data = img.get_fdata()
        cont_data = cont_pred.get_fdata()/1e5
        # see if anywhere in the continuous prediction is greater than 0.9 where img==1
        if not np.any(cont_data[unscaled_data>0.5] > 0.99):
            print(f'{file} does not have high confidence')
            continue
        else:
            print(f'{file} has high confidence')

        data = torch.tensor(unscaled_data)
        pixdim = img.header['pixdim'][1:4]
        target_spacing = 1
        scale_factor = [pixdim[0]/target_spacing, pixdim[1]/target_spacing, pixdim[2]/target_spacing]
        data = torch.nn.functional.interpolate(data.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='trilinear', align_corners=False)
        # round and convert to numpy
        data = torch.round(data)
        data = data.squeeze().numpy().astype(np.uint8)
        labels, n = scipy.ndimage.label(data)
        max_area = 0
        for i in range(1, n+1):
            area = np.sum(labels == i)
            if area > max_area:
                max_area = area
        if max_area < 14137:
            shutil.copy(os.path.join(homepath, file), os.path.join(small_pred_path, file))
            print(f'Copied {file} to {small_pred_path}')
            # find the 500mmth index by dividing 500 by vol of pixdim, sorting the continuous probabilities where data>0.5, and taking that index
            index = 500/(pixdim[0]*pixdim[1]*pixdim[2])
            sorted_conf = np.sort(cont_data[unscaled_data>0.5])
            print(f'500mmth index: {sorted_conf[int(index)]}, {sorted_conf[-int(index)]}')
            print(max_area)
            equiv_diam_cm = ((3*max_area)/(4*np.pi))**(1/3)*0.2
            print(f'Equiv diameter: {equiv_diam_cm}')
            print()
        else:
            print(f'{file} is too large ({max_area})')

