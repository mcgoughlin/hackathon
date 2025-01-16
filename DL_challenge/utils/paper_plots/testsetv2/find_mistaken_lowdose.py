import os
import pandas as pd
import numpy as np

full_dose = False

meta_folder = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/lowdose_ROC'
sub_folders = [os.path.join(meta_folder, f,'predictions.csv') for f in os.listdir(meta_folder) if os.path.isdir(os.path.join(meta_folder, f))]
names_sub_folders = [float(f.split('/')[-2]) for f in sub_folders]

# sort sub_folders by names_sub_folders
sub_folders = [x for _,x in sorted(zip(names_sub_folders,sub_folders))]
names_sub_folders = sorted(names_sub_folders)

highsens_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/lowdose_test/high_sens/predictions.csv'
highspec_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/lowdose_test/high_spec/predictions.csv'
highacc_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/lowdose_test/high_acc/predictions.csv'
iztok_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/iztokLD.csv'
cathal_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/cathalLD.csv'
truth_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internaltest_metadata.csv'
hania_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/haniaLD.csv'

# find the StudyID where cancer was missed by each radiologist that were found by the AI

# read in the AIpredictions
highacc_pred = pd.read_csv(highacc_pred_csv_fp)

# read in the radiologist predictions
iztok_pred = pd.read_csv(iztok_pred_csv_fp)
cathal_pred = pd.read_csv(cathal_pred_csv_fp)
hania_pred = pd.read_csv(hania_pred_csv_fp)

# read in the truth
truth = pd.read_csv(truth_csv_fp)

# find the StudyID where cancer was missed by each radiologist that were found by the AI
# for each kidney, if the AI found a cancer, and the radiologist did not, then record the StudyID

# convert radiologists in to one positive or negative using a majority vote
# compare the AI to the radiologist single maj. vote, compile all cases into the lists below
# posneg - AI found cancer, radiologist did not
# negneg - neither AI nor radiologist found cancer
# pospos - both AI and radiologist found cancer
# negpos - AI did not find cancer, radiologist did

#ignore where there is no cancer
posneg = []
negneg = []
pospos = []
negpos = []
for i,row in highacc_pred.iterrows():
    string = ''
    truth_row = truth[truth['StudyID'] == row['StudyID']]
    hania_row = hania_pred[hania_pred['StudyID'] == row['StudyID']]
    iztok_row = iztok_pred[iztok_pred['StudyID'] == row['StudyID']]
    cathal_row = cathal_pred[cathal_pred['StudyID'] == row['StudyID']]

    if (truth_row['RightCancer'].values[0] == 0):
        string += 'R '
    if (truth_row['LeftCancer'].values[0] == 0):
        string += 'L '

    #find the radiologist majority vote
    rightcount, leftcount = 0,0

    if iztok_row['RightCancer'].values[0] == 0:
        rightcount += 1
    if cathal_row['RightCancer'].values[0] == 0:
        rightcount += 1
    if hania_row['RightCancer'].values[0] == 0:
        rightcount += 1
    if iztok_row['LeftCancer'].values[0] == 0:
        leftcount += 1
    if cathal_row['LeftCancer'].values[0] == 0:
        leftcount += 1
    if hania_row['LeftCancer'].values[0] == 0:
        leftcount += 1


    rightcancer_rad = rightcount >= 2
    leftcancer_rad = leftcount >= 2

    print('ID:',row['StudyID'])
    print('rightcancer_rad:',rightcancer_rad)
    print('leftcancer_rad:',leftcancer_rad)

    #find the AI prediction
    rightcancer_ai = row['RightCancer'] == 0
    leftcancer_ai = row['LeftCancer'] == 0

    print('rightcancer_ai:',rightcancer_ai)
    print('leftcancer_ai:',leftcancer_ai)
    print()

    if rightcancer_ai and not rightcancer_rad:
        posneg.append(string + 'aiR '+ row['StudyID'])
    elif not rightcancer_ai and not rightcancer_rad:
        negneg.append(string + row['StudyID'])
    elif rightcancer_ai and rightcancer_rad:
        pospos.append(string + 'aiR radR '+ row['StudyID'])
    elif not rightcancer_ai and rightcancer_rad:
        negpos.append(string+ 'radR '+ row['StudyID'])

    if leftcancer_ai and not leftcancer_rad:
        posneg.append(string + 'aiL ' + row['StudyID'])
    elif not leftcancer_ai and not leftcancer_rad:
        negneg.append(string + row['StudyID'])
    elif leftcancer_ai and leftcancer_rad:
        pospos.append(string + 'aiL radL '+ row['StudyID'])
    elif not leftcancer_ai and leftcancer_rad:
        negpos.append(string + 'radL '+ row['StudyID'])


print('posneg:',posneg)
print('negneg:',negneg)
print('pospos:',pospos)
print('negpos:',negpos)
print('Length of all:',len(posneg)+len(negneg)+len(pospos)+len(negpos))
