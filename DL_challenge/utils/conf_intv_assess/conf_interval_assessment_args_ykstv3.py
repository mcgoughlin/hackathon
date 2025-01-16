import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from skimage.measure import label
import pandas as pd
import sys
import json

new_fold = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/ROCs/v3_files'
preds = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/ROCs/ykst_ROCv3'

# cancers : 10010791
# 10110488
# 13410964
# 16210581
# 16810841
# 17510823
# 17710851
# 18710093
# 20310997
# 20811235
# 21111898
# 21510856

# masses:
# 10410316
# 12810135
# 14410556
# 16410011
# 17610770
# 19110671
# 20010695

# bosniak 2F/3/4 cysts:
# 10011277
# 10610445
# 11111232
# 13110104
# 13411326
# 14410021
# 15210437
# 15711427
# 16711716
# 17211466
# 17211929
# 17310314
# 18610586
# 20310241
# 21111173

# oncocytoma:
# 14911662
# 18710764
# 20510158
# 20710211
# 21110062

# uretic cancer:
# 20410693

positives_lb = {'10010791':'cancer', '10110488':'cancer', '13410964':'cancer', '16210581':'cancer', '16810841':'cancer',
                '17510823':'cancer', '17710851':'cancer', '18710093':'cancer', '20310997':'cancer', '20811235':'cancer',
                '21111898':'cancer', '21510856':'cancer', '10410316':'mass', '12810135':'mass', '14410556':'mass',
                '16410011':'mass', '17610770':'mass', '19110671':'mass', '20010695':'mass', '10011277':'cyst',
                '10610445':'cyst', '11111232':'cyst', '13110104':'cyst', '13411326':'cyst', '14410021':'cyst',
                '15210437':'cyst', '15711427':'cyst', '16711716':'cyst', '17211466':'cyst', '17211929':'cyst',
                '17310314':'cyst', '18610586':'cyst', '20310241':'cyst', '21111173':'cyst', '14911662':'oncocytoma',
                '18710764':'oncocytoma', '20510158':'oncocytoma', '20710211':'oncocytoma', '21110062':'oncocytoma',
                '20410693':'uretic_cancer'}

if not os.path.exists(new_fold):
    os.makedirs(new_fold)
png_path = os.path.join(new_fold, 'YKST_ROC.png')

per_patient_confidences = {}

confidences = [fold for fold in os.listdir(preds) if os.path.isdir(os.path.join(preds, fold))]

results = [[[1,0],[1,0],[1,0]]]

for x,conf in enumerate(confidences):
    print(conf)
    print('Confidence: {}'.format(conf))
    # need to store tp, tn, fp, fn for each of the following groups:
    # cancers: just cancers (cancer, uretic cancer)
    # solid masses: ust cancers (cancer, uretic cancer) , masses, and oncocytomas
    # all

    case_wise = {'cancers': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                    'solid_masses': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                    'all': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}

    conf_path = os.path.join(preds, str(conf))
    positives_pred = os.path.join(conf_path, 'positive.csv')
    negatives_pred = os.path.join(conf_path, 'negative.csv')
    # dfs do not have headers - filenames are first column
    positives_df = pd.read_csv(positives_pred, header=None)
    negatives_df = pd.read_csv(negatives_pred, header=None)

    # get the filenames
    positives = positives_df[0].values
    negatives = negatives_df[0].values

    # crop the first 8 characters to get the filename
    positives = [str(file[:8]) for file in positives]
    negatives = [str(file[:8]) for file in negatives]

    # get all filenames
    files = np.concatenate([positives, negatives])
    pos_count = 0
    for file in files:
        filename = str(file[:8])
        positive = filename in positives

        if positive:
            pos_count+=1
            if filename in positives_lb:
                case_wise['all']['tp'] += 1
                if positives_lb[filename] == 'cancer' or positives_lb[filename] == 'uretic_cancer':
                    case_wise['cancers']['tp'] += 1
                    case_wise['solid_masses']['tp'] += 1
                elif positives_lb[filename] == 'mass' or positives_lb[filename] == 'oncocytoma':
                    case_wise['solid_masses']['tp'] += 1
                    case_wise['cancers']['fp'] += 1
                else:
                    case_wise['cancers']['fp'] += 1
                    case_wise['solid_masses']['fp'] += 1
            else:
                case_wise['all']['fp'] += 1
                case_wise['cancers']['fp'] += 1
                case_wise['solid_masses']['fp'] += 1
            if filename in per_patient_confidences:
                per_patient_confidences[filename] = conf
        else:
            if filename in positives_lb:
                case_wise['all']['fn'] += 1
                if positives_lb[filename] == 'cancer' or positives_lb[filename] == 'uretic_cancer':
                    case_wise['cancers']['fn'] += 1
                    case_wise['solid_masses']['fn'] += 1
                elif positives_lb[filename] == 'mass' or positives_lb[filename] == 'oncocytoma':
                    print(filename)
                    case_wise['solid_masses']['fn'] += 1
                    case_wise['cancers']['tn'] += 1
                else:
                    case_wise['cancers']['tn'] += 1
                    case_wise['solid_masses']['tn'] += 1
            else:
                case_wise['all']['tn'] += 1
                case_wise['cancers']['tn'] += 1
                case_wise['solid_masses']['tn'] += 1
    # calculate sensitivity and specificity for each group
    conf_results = []
    for group in case_wise:
        tp = case_wise[group]['tp']
        tn = case_wise[group]['tn']
        fp = case_wise[group]['fp']
        fn = case_wise[group]['fn']
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        conf_results.append([sensitivity, specificity])
    results.append(conf_results)

    print(conf_results[0])
    print()

#sort the order the results by the confs in list confidences
results = np.array(results)
results = results[1:]
results = results[np.argsort(confidences)]



# plot the ROC curves for each group
plot_results = np.concatenate([results,[[[0,1],[0,1],[0,1]]]],axis=0)
plot_results = np.concatenate([[[[1,0],[1,0],[1,0]]],plot_results],axis=0)

plot_results*=100
for i in range(len(confidences)):
    #print the mass ROCs and cofndience
    print('Confidence: {}'.format(confidences[i]))
    print('Cancers: Sensitivity: {:.2f} Specificity: {:.2f}'.format(plot_results[i,0,0], plot_results[i,0,1]))
    print('Masses: Sensitivity: {:.2f} Specificity: {:.2f}'.format(plot_results[i,1,0], plot_results[i,1,1]))
    print()

cancer_auc = np.trapz(plot_results[:,0,1], plot_results[:,0,0])
mass_auc = np.trapz(plot_results[:,1,1], plot_results[:,1,0])
all_auc = np.trapz(plot_results[:,2,1], plot_results[:,2,0])


fig = plt.figure(figsize=(8,7))
# choose colour not used anywhere else (black, red, blue)
plt.plot(plot_results[:,0,1], plot_results[:,0,0], label='Cancers (AUC {:.3f})'.format(cancer_auc/-10000),
         c = 'k', linewidth=2)
plt.plot(plot_results[:,1,1], plot_results[:,1,0], label='All solid lesions (AUC {:.3f})'.format(mass_auc/-10000),
            c = 'k', linewidth=2, linestyle=':')

max_spec = np.max(plot_results[:,0,1][plot_results[:,0,0]==100])
plt.plot([max_spec,max_spec],[-2,100],color='gray',linestyle='--')
# annotate this vertical line with 'High-Risk Cancer Prioritisation'
plt.text(max_spec - 3, 40, 'All cancers detected', rotation=90, verticalalignment='center',
            fontsize=12, fontdict={'family':'Helvetica'})


okspec = 80.8
plt.plot([okspec,okspec],[-2,100],color='gray',linestyle='--')
# annotate this vertical line with 'All Masses over 2cm detected'
plt.text(okspec - 3, 40, 'High-accuracy point: All solid lesions over 1.8cm detected', rotation=90,
         verticalalignment='center', fontsize=12, fontdict={'family':'Helvetica'})
lt = plt.legend(fontsize=16,ncol=1,loc='lower left')
#change lt font to helvetica
for text in lt.get_texts():
    text.set_fontname('Helvetica')
    text.set_fontsize(14)
# set the x and y axis limits
plt.xlim([-2, 102])
plt.ylim([-2, 102])

#plt minor ticks
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.25', color='black')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.xlabel('Specificity / %', fontsize=16,fontdict={'family':'Helvetica'})
plt.ylabel('Sensitivity / %', fontsize=16,fontdict={'family':'Helvetica'})
plt.tight_layout()
plt.savefig(png_path,dpi=300)
plt.show()
plt.close()

# save the ROC data for each group separately
save_results = np.concatenate([results,[[[0,1],[0,1],[0,1]]]],axis=0)
cancers_rocdf = pd.DataFrame(save_results[:,0], columns=['Sensitivity', 'Specificity'])
cancers_rocdf.to_csv(os.path.join(new_fold, 'cancers_ROC.csv'))

masses_rocdf = pd.DataFrame(save_results[:,1], columns=['Sensitivity', 'Specificity'])
masses_rocdf.to_csv(os.path.join(new_fold, 'masses_ROC.csv'))

all_rocdf = pd.DataFrame(save_results[:,2], columns=['Sensitivity', 'Specificity'])
all_rocdf.to_csv(os.path.join(new_fold, 'all_ROC.csv'))