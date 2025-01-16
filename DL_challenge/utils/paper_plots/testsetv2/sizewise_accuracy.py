import numpy as np
import pandas as pd
import os
import scipy.ndimage as ndimage
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

home = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/lowdose_ROC'
sub_folders = [f for f in os.listdir(home) if os.path.isdir(os.path.join(home, f))]
sub_folders = [os.path.join(home, f) for f in sub_folders]
confidences = [float(f.split('/')[-1]) for f in sub_folders]

# if rightcancer is positive in subfolder 0.1, but negative in 0.2, then the confidence is 0.1
percase_confidences = {}

label_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/full_labels.csv'
labels = pd.read_csv(label_loc)
#loop through patients and fill percase_confidences with correct labels
for patient in labels.StudyID:
    percase_confidences[patient] = {'RightCancer':0,'LeftCancer':0,'RLabel':labels.loc[labels.StudyID==patient,'RightCancer'].values[0],'LLabel':labels.loc[labels.StudyID==patient,'LeftCancer'].values[0],'RCystic':labels.loc[labels.StudyID==patient,'RightCystic'].values[0],'LCystic':labels.loc[labels.StudyID==patient,'LeftCystic'].values[0],'SusMass':labels.loc
    [labels.StudyID==patient,'SusMass'].values[0], 'size':0,'exophytic':0}

truth_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internaltest_metadata.csv'
iztok_predfulldose_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/iztokLD.csv'
hania_predfulldose_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/haniaLD.csv'
cathal_predfulldose_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/cathalLD.csv'

#add the full dose predictions to the labels df
iztok_pred = pd.read_csv(iztok_predfulldose_csv_fp)
hania_pred = pd.read_csv(hania_predfulldose_csv_fp)
cathal_pred = pd.read_csv(cathal_predfulldose_csv_fp)

#append hania RightCancer and LeftCancer to labels via hania_RightCancer and hania_LeftCancer. append based on StudyID
labels = labels.merge(hania_pred[['StudyID','RightCancer','LeftCancer']],on='StudyID',how='left',suffixes=('','_hania'))
#append iztok RightCancer and LeftCancer to labels via iztok_RightCancer and iztok_LeftCancer. append based on StudyID
labels = labels.merge(iztok_pred[['StudyID','RightCancer','LeftCancer']],on='StudyID',how='left',suffixes=('','_iztok'))
#append cathal RightCancer and LeftCancer to labels via cathal_RightCancer and cathal_LeftCancer. append based on StudyID
labels = labels.merge(cathal_pred[['StudyID','RightCancer','LeftCancer']],on='StudyID',how='left',suffixes=('','_cathal'))

import nibabel as nib
seg_loc = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/test_set_v2/labels'
kidney_seg_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/kidney_segs/'
for key in percase_confidences.keys():
    # find case name fr0m labels
    case = labels.loc[labels.StudyID==key,'OriginalID'].values[0]
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

#loop through confidences from lowest to highest
for confidence,folder in zip(confidences,sub_folders):
    #read in predictions
    predictions = pd.read_csv(os.path.join(folder,'predictions.csv'))
    #loop through each patient
    for patient in predictions.StudyID:
        #if the right kidney is positive, and the confidence is not already recorded, record the confidence

        if predictions.loc[predictions.StudyID==patient,'RightCancer'].values[0] == 1 and percase_confidences[patient]['RightCancer'] < confidence:
            percase_confidences[patient]['RightCancer'] = confidence
        #if the left kidney is positive, and the confidence is not already recorded, record the confidence
        if predictions.loc[predictions.StudyID==patient,'LeftCancer'].values[0] == 1 and percase_confidences[patient]['LeftCancer'] < confidence:
            percase_confidences[patient]['LeftCancer'] = confidence



percase_confidences = pd.DataFrame(percase_confidences).T
# save in home
# percase_confidences.to_csv(os.path.join(home,'percase_confidences.csv'))
# convert df into a list of single observations, with columns: ID, Kidney, Confidence, Label, Cystic, SusMass,size
observations = []
for patient in percase_confidences.index:
    observations.append([patient,'Right',percase_confidences.loc[patient,'RightCancer'],percase_confidences.loc[patient,'RLabel'],percase_confidences.loc[patient,'RCystic'],percase_confidences.loc[patient,'SusMass'],percase_confidences.loc[patient,'size']])
    observations.append([patient,'Left',percase_confidences.loc[patient,'LeftCancer'],percase_confidences.loc[patient,'LLabel'],percase_confidences.loc[patient,'LCystic'],percase_confidences.loc[patient,'SusMass'],percase_confidences.loc[patient,'size']])


observations = pd.DataFrame(observations,columns=['ID','Kidney','Confidence','Label','Cystic','SusMass','Size'])
observations.to_csv(os.path.join(home,'observations.csv'))

percase_confidences['StudyID'] = percase_confidences.index.values
#merge rad predictions with percase_confidences
percase_confidences = percase_confidences.merge(labels[['StudyID','RightCancer_hania','LeftCancer_hania','RightCancer_iztok','LeftCancer_iztok','RightCancer_cathal','LeftCancer_cathal']],on='StudyID',how='left')

#convert columns RightCancer and LeftCancer to AIRight and AILeft, with 1 if >0.702, 0 otherwise
percase_confidences['AIRight'] = (percase_confidences['RightCancer']>0.702).astype(int)
percase_confidences['AILeft'] = (percase_confidences['LeftCancer']>0.702).astype(int)
#delete columns RightCancer and LeftCancer
percase_confidences.drop(columns=['RightCancer','LeftCancer'],inplace=True)

#extract all cases where size < 14137
small = percase_confidences.loc[percase_confidences['size'] < 14137]
large = percase_confidences.loc[percase_confidences['size'] >= 14137]

print('small')

#find sens and spec for AI across both kidneys
tp = np.sum(np.logical_and(small['RLabel'],small['AIRight'])) + np.sum(np.logical_and(small['LLabel'],small['AILeft']))
tn = np.sum(np.logical_and(np.logical_not(small['RLabel']),np.logical_not(small['AIRight']))) + np.sum(np.logical_and(np.logical_not(small['LLabel']),np.logical_not(small['AILeft'])))
fp = np.sum(np.logical_and(np.logical_not(small['RLabel']),small['AIRight'])) + np.sum(np.logical_and(np.logical_not(small['LLabel']),small['AILeft']))
fn = np.sum(np.logical_and(small['RLabel'],np.logical_not(small['AIRight']))) + np.sum(np.logical_and(small['LLabel'],np.logical_not(small['AILeft'])))
sens = tp/(tp+fn)
spec = tn/(tn+fp)
print(f'AISensitivity: {100*sens:.2f}%')
print(f'Specificity: {100*spec:.2f}%')

#do the same for radiologists
tp = np.sum(np.logical_and(small['RLabel'],small['RightCancer_hania'])) + np.sum(np.logical_and(small['LLabel'],small['LeftCancer_hania']))
tn = np.sum(np.logical_and(np.logical_not(small['RLabel']),np.logical_not(small['RightCancer_hania']))) + np.sum(np.logical_and(np.logical_not(small['LLabel']),np.logical_not(small['LeftCancer_hania'])))
fp = np.sum(np.logical_and(np.logical_not(small['RLabel']),small['RightCancer_hania'])) + np.sum(np.logical_and(np.logical_not(small['LLabel']),small['LeftCancer_hania']))
fn = np.sum(np.logical_and(small['RLabel'],np.logical_not(small['RightCancer_hania'])))+ np.sum(np.logical_and(small['LLabel'],np.logical_not(small['LeftCancer_hania'])))
sens = tp/(tp+fn)
spec = tn/(tn+fp)
print(f'HaniaSensitivity: {100*sens:.2f}%')
print(f'Specificity: {100*spec:.2f}%')

tp = np.sum(np.logical_and(small['RLabel'],small['RightCancer_iztok'])) + np.sum(np.logical_and(small['LLabel'],small['LeftCancer_iztok']))
tn = np.sum(np.logical_and(np.logical_not(small['RLabel']),np.logical_not(small['RightCancer_iztok']))) + np.sum(np.logical_and(np.logical_not(small['LLabel']),np.logical_not(small['LeftCancer_iztok'])))
fp = np.sum(np.logical_and(np.logical_not(small['RLabel']),small['RightCancer_iztok'])) + np.sum(np.logical_and(np.logical_not(small['LLabel']),small['LeftCancer_iztok']))
fn = np.sum(np.logical_and(small['RLabel'],np.logical_not(small['RightCancer_iztok'])))+ np.sum(np.logical_and(small['LLabel'],np.logical_not(small['LeftCancer_iztok'])))
sens = tp/(tp+fn)
spec = tn/(tn+fp)
print(f'IztokSensitivity: {100*sens:.2f}%')
print(f'Specificity: {100*spec:.2f}%')

tp = np.sum(np.logical_and(small['RLabel'],small['RightCancer_cathal'])) + np.sum(np.logical_and(small['LLabel'],small['LeftCancer_cathal']))
tn = np.sum(np.logical_and(np.logical_not(small['RLabel']),np.logical_not(small['RightCancer_cathal']))) + np.sum(np.logical_and(np.logical_not(small['LLabel']),np.logical_not(small['LeftCancer_cathal'])))
fp = np.sum(np.logical_and(np.logical_not(small['RLabel']),small['RightCancer_cathal'])) + np.sum(np.logical_and(np.logical_not(small['LLabel']),small['LeftCancer_cathal']))
fn = np.sum(np.logical_and(small['RLabel'],np.logical_not(small['RightCancer_cathal'])))+ np.sum(np.logical_and(small['LLabel'],np.logical_not(small['LeftCancer_cathal'])))
sens = tp/(tp+fn)
spec = tn/(tn+fp)
print(f'CathalSensitivity: {100*sens:.2f}%')
print(f'Specificity: {100*spec:.2f}%')
print()
#repeat for large
print('large')
tp = np.sum(np.logical_and(large['RLabel'],large['AIRight'])) + np.sum(np.logical_and(large['LLabel'],large['AILeft']))
tn = np.sum(np.logical_and(np.logical_not(large['RLabel']),np.logical_not(large['AIRight'])))+ np.sum(np.logical_and(np.logical_not(large['LLabel']),np.logical_not(large['AILeft'])))
fp = np.sum(np.logical_and(np.logical_not(large['RLabel']),large['AIRight'])) + np.sum(np.logical_and(np.logical_not(large['LLabel']),large['AILeft']))
fn = np.sum(np.logical_and(large['RLabel'],np.logical_not(large['AIRight'])))+ np.sum(np.logical_and(large['LLabel'],np.logical_not(large['AILeft'])))
sens = tp/(tp+fn)
spec = tn/(tn+fp)
print(f'AISensitivity: {100*sens:.2f}%')
print(f'Specificity: {100*spec:.2f}%')

tp = np.sum(np.logical_and(large['RLabel'],large['RightCancer_hania'])) + np.sum(np.logical_and(large['LLabel'],large['LeftCancer_hania']))
tn = np.sum(np.logical_and(np.logical_not(large['RLabel']),np.logical_not(large['RightCancer_hania']))) + np.sum(np.logical_and(np.logical_not(large['LLabel']),np.logical_not(large['LeftCancer_hania'])))
fp = np.sum(np.logical_and(np.logical_not(large['RLabel']),large['RightCancer_hania'])) + np.sum(np.logical_and(np.logical_not(large['LLabel']),large['LeftCancer_hania']))
fn = np.sum(np.logical_and(large['RLabel'],np.logical_not(large['RightCancer_hania'])))+ np.sum(np.logical_and(large['LLabel'],np.logical_not(large['LeftCancer_hania'])))
sens = tp/(tp+fn)
spec = tn/(tn+fp)
print(f'HaniaSensitivity: {100*sens:.2f}%')
print(f'Specificity: {100*spec:.2f}%')

tp = np.sum(np.logical_and(large['RLabel'],large['RightCancer_iztok'])) + np.sum(np.logical_and(large['LLabel'],large['LeftCancer_iztok']))
tn = np.sum(np.logical_and(np.logical_not(large['RLabel']),np.logical_not(large['RightCancer_iztok']))) + np.sum(np.logical_and(np.logical_not(large['LLabel']),np.logical_not(large['LeftCancer_iztok'])))
fp = np.sum(np.logical_and(np.logical_not(large['RLabel']),large['RightCancer_iztok'])) + np.sum(np.logical_and(np.logical_not(large['LLabel']),large['LeftCancer_iztok']))
fn = np.sum(np.logical_and(large['RLabel'],np.logical_not(large['RightCancer_iztok'])))+ np.sum(np.logical_and(large['LLabel'],np.logical_not(large['LeftCancer_iztok'])))
sens = tp/(tp+fn)
spec = tn/(tn+fp)
print(f'IztokSensitivity: {100*sens:.2f}%')
print(f'Specificity: {100*spec:.2f}%')

tp = np.sum(np.logical_and(large['RLabel'],large['RightCancer_cathal'])) + np.sum(np.logical_and(large['LLabel'],large['LeftCancer_cathal']))
tn = np.sum(np.logical_and(np.logical_not(large['RLabel']),np.logical_not(large['RightCancer_cathal'])))+ np.sum(np.logical_and(np.logical_not(large['LLabel']),np.logical_not(large['LeftCancer_cathal'])))
fp = np.sum(np.logical_and(np.logical_not(large['RLabel']),large['RightCancer_cathal'])) + np.sum(np.logical_and(np.logical_not(large['LLabel']),large['LeftCancer_cathal']))
fn = np.sum(np.logical_and(large['RLabel'],np.logical_not(large['RightCancer_cathal'])))+ np.sum(np.logical_and(large['LLabel'],np.logical_not(large['LeftCancer_cathal'])))
sens = tp/(tp+fn)
spec = tn/(tn+fp)
print(f'CathalSensitivity: {100*sens:.2f}%')
print(f'Specificity: {100*spec:.2f}%')