import nibabel as nib
import numpy as np
import os
import torch
from scipy.ndimage.morphology import binary_fill_holes
import scipy.ndimage as spim
from skimage.measure import regionprops
from KCD.Detection.Preprocessing.ObjectFiles import object_creation_utils as ocu
from skimage.segmentation import watershed

# we wan to find the optimum confidence threshold for determining the presence of cancerous voxel labels,
# and the optimum size threshold for determining the presence of a cancerous region
# we will do this by finding the threshold that maximises the dice score on a dataset

path = '/Users/mcgoug01/Downloads/valid_v4'
cancer_infp = os.path.join(path, 'cross_validation_continuous')
cancer_gt = os.path.join(path, 'labels')
print(path)

confidence_thresholds = np.append(np.arange(0,0.1,0.01),np.arange(0.1, 0.9, 0.05))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.9,0.98,0.02))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.98,1.005,0.005))
vol = 500
print(confidence_thresholds)

results =[]
percase_results =[]

#only load files that end in .nii.gz
for conf in confidence_thresholds:
    conf_results = []
    tp, fp, fn, tn = 0, 0, 0, 0
    for file in [path for path in os.listdir(cancer_infp) if path.endswith('.nii.gz')]:
        patient_results = [0,0,0,0] #tp, fp, fn, tn
        try:
            cancer_inf = nib.load(os.path.join(cancer_infp, file))
            cancer_label = nib.load(os.path.join(cancer_gt, file))
        except:
            continue
        # assert all images have same shape
        assert cancer_inf.shape == cancer_label.shape
        #find spacing
        spacing = cancer_inf.header['pixdim'][1:4]

        #resize all to 2mm spacing using torch
        try:
            cancer_inf = torch.from_numpy(cancer_inf.get_fdata()).unsqueeze(0).unsqueeze(0)
            cancer_label = torch.from_numpy(cancer_label.get_fdata()).unsqueeze(0).unsqueeze(0)
        except:
            continue
        cancer_inf = torch.nn.functional.interpolate(cancer_inf, scale_factor=tuple(spacing/2), mode='trilinear')
        cancer_label = torch.nn.functional.interpolate(cancer_label, scale_factor=tuple(spacing/2), mode='trilinear', align_corners=False)

        #find cancerous voxels
        cancer_voxels = cancer_inf > conf
        # fill holes in cancerous voxels
        cancer_voxels = binary_fill_holes(cancer_voxels)
        #filter out regions with less than vol voxels using regionprops and spim
        cancer_voxels = spim.label(cancer_voxels.squeeze())[0]
        cancer_regions = np.array([region for region in regionprops(cancer_voxels) if region.area > vol])

        # reconstruct cancerous voxels from regions over vol
        cancer_voxels = np.zeros_like(cancer_voxels)
        for region in cancer_regions:
            cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]] = 2

        # colour kidney labels in 2 if cancerous, 1 if not
        travel_mask1 = cancer_label.squeeze().numpy().copy()
        #set 0 values to -1, set 1 values to 0 in travel mask
        travel_mask1[travel_mask1==0] = -1
        travel_mask1[travel_mask1==1] = 0
        cancer_kidney_labels = watershed(travel_mask1, markers=(cancer_label.numpy().squeeze()==2).astype(int),
                                  mask = (cancer_label.numpy().squeeze()>0).astype(int))
        cancer_kidney_labels[cancer_kidney_labels==1] = 2
        cancer_kidney_labels[(cancer_kidney_labels==0) & (cancer_label.numpy().squeeze()>0)] = 1

        # reconstruct labels voxels from cancer_kidney_labels, removing tiny regions
        label_voxels = np.zeros_like(cancer_kidney_labels)
        for label in [1,2]:
            for region in regionprops(spim.label(cancer_kidney_labels==label)[0]):
                if region.area > 20:
                    label_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]] = label

        # loop through all cancer predicted regions without a corresponding kidney label,
        # and add a false positive
        for region in regionprops(spim.label(cancer_voxels == 2)[0]):
            kid_reg = cancer_label[0,0,region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]]
            if kid_reg.max() == 0:
                fp += 1
                patient_results[1] += 1

        # loop through all cancerous labels - checking if true or false positive
        for region in regionprops(spim.label(label_voxels==2)[0]):
            prediction_region = cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]]
            if prediction_region.max() <2:
                fn+=1
                patient_results[2] += 1
            else:
                tp+=1
                patient_results[0] += 1

        # loop through all healthy labels - checking if true or false negative
        for region in regionprops(spim.label(label_voxels==1)[0]):
            prediction_region = cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]]
            if prediction_region.max() <2:
                tn+=1
                patient_results[3] += 1
            else:
                fp+=1
                patient_results[1] += 1
        conf_results.append(patient_results)
    percase_results.append(conf_results)
    entry = {'confidence_threshold': conf, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    sensitivity = tp/(tp+fn+1e-6)
    specificity = tn/(tn+fp+1e-6)
    precision = tp/(tp+fp+1e-6)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    entry['sensitivity'] = sensitivity
    entry['specificity'] = specificity
    entry['precision'] = precision
    entry['accuracy'] = accuracy
    entry['dice'] = 2*tp/(2*tp+fp+fn+1e-6)

    print(entry)
    results.append(entry)

percase_results = np.array(percase_results)
# save results to npy
np.save(os.path.join(path, 'results_confwise.npy'), percase_results)

import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
df = pd.DataFrame(results)
df.to_csv(os.path.join(path, 'results_all_validation.csv'))

path_to_results = os.path.join(path, 'results_all_validation.csv')
df = pd.read_csv(path_to_results)
plt.figure()

fpr = 100*(1-df['specificity'])
sens = 100*df['sensitivity']
fpr = np.append(fpr,0)
sens = np.append(sens,0)

AUC = np.abs(np.trapz(sens,fpr)/1e4)
print(AUC)

#append (0,0) and (100,100) to x and y

# plot ROC curve
plt.plot(fpr,sens,label='{} (area = {:.3f})'.format(vol,AUC))
plt.ylabel('Sensitivity (%)')
plt.xlabel('1 - Specificity (%)')
plt.title('2-Stage Segmentation-based Detection Validation ROC')
plt.legend(loc="lower right")
plt.savefig(os.path.join(path, 'roc_validation.png'))
plt.show()