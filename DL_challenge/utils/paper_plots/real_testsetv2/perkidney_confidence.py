import numpy as np
import pandas as pd
import os
import scipy.ndimage as ndimage
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

#want to find equivalent confidence at which each kidney in each patient goes from predicted positive to negative.
# My folder structure is:

# /home/
#     - /0.01/
#           - /predictions.csv
#     - /0.02/
#           - /predictions.csv
#  ...

# each predictions.csv contains an identical set of rows and columns, with data either '1' or '0' for positive, or negative, respectively
# each column is right/left kidney cancer, named 'RightCancer' and 'LeftCancer'
# each row corresponds to each patient, named ID1, ID2, etc.

#find the highest confidence that shows '1' in each kidney.

home = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions_real/fulldose_ROC'
sub_folders = [f for f in os.listdir(home) if os.path.isdir(os.path.join(home, f))]
sub_folders = [os.path.join(home, f) for f in sub_folders]
confidences = [float(f.split('/')[-1]) for f in sub_folders]

# if rightcancer is positive in subfolder 0.1, but negative in 0.2, then the confidence is 0.1
percase_confidences = {}

label_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internaltest_metadata_real.csv'
labels = pd.read_csv(label_loc)
#loop through patients and fill percase_confidences with correct labels
for patient in labels.OriginalID:
    percase_confidences[patient] = {'RightCancer':0,'LeftCancer':0,'RLabel':labels.loc[labels.OriginalID==patient,'RightCancer'].values[0],'LLabel':labels.loc[labels.OriginalID==patient,'LeftCancer'].values[0],'size':0}

import nibabel as nib
seg_loc = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/test_set_v2/labels'
kidney_seg_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/kidney_segs/'
for key in percase_confidences.keys():
    # find case name fr0m labels
    case = key
    # if case.lower does not begin with kits or rcc, skip
    if not case.lower().startswith('kits') and not case.lower().startswith('rcc'):
        continue

    # if case begins with kits in seg_loc
    if case.lower().startswith('kits'):
        case = case.lower().replace('kits-','KiTS-')
        seg_n = nib.load(os.path.join(seg_loc,case))
        seg = (seg_n.get_fdata()==2).astype(np.uint8)
        seg = np.swapaxes(seg,0,2)
        vox_vol = np.prod(seg_n.header.get_zooms())
        percase_confidences[key]['size'] = np.sum(seg)*vox_vol
    # if case begins with rcc in seg_loc
    elif case.lower().startswith('rcc'):
        case = case.lower().replace('rcc_','Rcc_')
        seg_n = nib.load(os.path.join(seg_loc,case))
        seg = (seg_n.get_fdata()==1).astype(np.uint8)
        vox_vol = np.prod(seg_n.header.get_zooms())
        percase_confidences[key]['size'] = np.sum(seg)*vox_vol
        #flip up down
        # seg = np.flip(seg,axis=0)
    else:
        continue
    #
    # kidney_seg = nib.load(os.path.join(kidney_seg_loc,key+'.nii.gz'))
    # kidney_seg = kidney_seg.get_fdata()
    # #resize kidney_seg to seg shape using torch.nn.functional.interpolate
    # kidney_seg = torch.tensor(kidney_seg).unsqueeze(0).unsqueeze(0).float()
    # seg = torch.tensor(seg).unsqueeze(0).unsqueeze(0).float()
    # kidney_seg = F.interpolate(kidney_seg, size=seg.shape[2:], mode='nearest').squeeze().numpy()
    # kidney_seg = kidney_seg.astype(np.uint8)
    # seg = seg.squeeze().numpy().astype(np.uint8)
    # #combine kidney seg and cancer seg
    # combined = np.zeros(kidney_seg.shape)
    # combined[kidney_seg==1] = 1
    # combined[seg==1] = 2
    # dilate = np.zeros(combined.shape, dtype=np.uint8)
    # dilate[combined == 2] = 1
    # dilate = ndimage.binary_dilation(dilate, iterations=1).astype(np.uint8)
    # combined[(dilate == 1) & (combined == 1)] = 2
    #
    # #find the largest cancer region and extract its segmentation
    # labels_indices, n = ndimage.label(combined == 2)
    # sizes = ndimage.sum(combined, labels_indices, range(n+1))
    # max_label = np.argmax(sizes) + 1
    # largest_cancer = np.zeros(combined.shape, dtype=np.uint8)
    # largest_cancer[labels_indices != max_label] = 0
    # largest_cancer[labels_indices == max_label] = 1
    # # extract volume, surface area, sphericity, largest diameter, and exophytic fraction
    # volume = np.sum(largest_cancer)
    # # scale volume to mm^3 and then to cm^3
    # contour = ndimage.binary_dilation(largest_cancer, iterations=1).astype(np.uint8) - largest_cancer
    # surface_area = np.sum(contour) * 2
    # # scale surface area to mm^2 and then to cm^2
    # sphericity = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surface_area
    #
    # # find the exophytic fraction
    # # exophytic fraction is ratio of contour that is within new_data==1
    # exophytic = 1 - np.sum(contour[combined == 1]) / np.sum(contour)
    # percase_confidences[key]['exophytic'] = exophytic
    # # plot the slice with the largest cancer region for both seg and cancer seg
    # #argmax of seg in 3rd dimension is the slice with the largest cancer region
    # slice = np.argmax([np.sum(largest_cancer[:,:,i]) for i in range(largest_cancer.shape[2])])
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(seg[:,:,slice],cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(kidney_seg[:,:,slice],cmap='gray')
    # plt.show(block=True)
    # print('Key:',key,'Case',case,'Shapes:',seg.shape,kidney_seg.shape)
    # print(percase_confidences[key])



#loop through confidences from lowest to highest
for confidence,folder in zip(confidences,sub_folders):
    #read in predictions
    predictions = pd.read_csv(os.path.join(folder,'predictions.csv'))
    #loop through each patient
    for patient in predictions.OriginalID:
        #if the right kidney is positive, and the confidence is not already recorded, record the confidence

        if predictions.loc[predictions.OriginalID==patient,'RightCancer'].values[0] == 1 and percase_confidences[patient]['RightCancer'] < confidence:
            percase_confidences[patient]['RightCancer'] = confidence
        #if the left kidney is positive, and the confidence is not already recorded, record the confidence
        if predictions.loc[predictions.OriginalID==patient,'LeftCancer'].values[0] == 1 and percase_confidences[patient]['LeftCancer'] < confidence:
            percase_confidences[patient]['LeftCancer'] = confidence

#convert to dataframe
print(percase_confidences)

percase_confidences = pd.DataFrame(percase_confidences).T
# save in home
# percase_confidences.to_csv(os.path.join(home,'percase_confidences.csv'))
# convert df into a list of single observations, with columns: ID, Kidney, Confidence, Label, Cystic, SusMass,size
observations = []
for patient in percase_confidences.index:
    observations.append([patient,'Right',percase_confidences.loc[patient,'RightCancer'],percase_confidences.loc[patient,'RLabel'],percase_confidences.loc[patient,'size']])
    observations.append([patient,'Left',percase_confidences.loc[patient,'LeftCancer'],percase_confidences.loc[patient,'LLabel'],percase_confidences.loc[patient,'size']])

observations = pd.DataFrame(observations,columns=['ID','Kidney','Confidence','Label','Size'])

#where label ==0, set size to 0
observations.loc[observations['Label']==0,'Size'] = 0
observations.to_csv(os.path.join(home,'observations.csv'))