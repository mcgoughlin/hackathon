import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from skimage.measure import label
import pandas as pd
import sys
import json

home = '/bask/projects/p/phwq4930-renal-canc/data/seg_data'
preds = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions_nii/masked_ykst/s2_[2 2 2]mm_cont'
new_fold = os.path.join(home, 'conv_intv', 'ykst_corrected_alt2')
save_loc = os.path.join(new_fold, 'conf_ROC.npy')
ppc_path = os.path.join(new_fold, 'conf_per_patient.npy')
csv_path = os.path.join(new_fold, 'conf_ROC.csv')
png_path = os.path.join(new_fold, 'conf_ROC.png')

if not os.path.exists(new_fold):
    os.makedirs(new_fold)

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

positives = {'10010791':'cancer', '10110488':'cancer', '13410964':'cancer', '16210581':'cancer', '16810841':'cancer',
                '17510823':'cancer', '17710851':'cancer', '18710093':'cancer', '20310997':'cancer', '20811235':'cancer',
                '21111898':'cancer', '21510856':'cancer', '10410316':'mass', '12810135':'mass', '14410556':'mass',
                '16410011':'mass', '17610770':'mass', '19110671':'mass', '20010695':'mass', '10011277':'cyst',
                '10610445':'cyst', '11111232':'cyst', '13110104':'cyst', '13411326':'cyst', '14410021':'cyst',
                '15210437':'cyst', '15711427':'cyst', '16711716':'cyst', '17211466':'cyst', '17211929':'cyst',
                '17310314':'cyst', '18610586':'cyst', '20310241':'cyst', '21111173':'cyst', '14911662':'oncocytoma',
                '18710764':'oncocytoma', '20510158':'oncocytoma', '20710211':'oncocytoma', '21110062':'oncocytoma',
                '20410693':'uretic_cancer'}



confidences = [99500,99600,99700,99800,99900,99950,99960,99970,99980,99990]
files = [f for f in os.listdir(preds) if f.endswith('.nii.gz')]
per_patient_confidences = {}

results = [[[1,0],[1,0],[1,0]]]

for x,conf in enumerate(confidences):
    # need to store tp, tn, fp, fn for each of the following groups:
    # cancers: just cancers (cancer, uretic cancer)
    # solid masses: ust cancers (cancer, uretic cancer) , masses, and oncocytomas
    # all

    case_wise = {'cancers': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                    'solid_masses': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                    'all': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}
    for y,file in enumerate(files):
        filename = str(file[:8])
        if not (filename in per_patient_confidences):
            per_patient_confidences[filename] = 0
        pred_nib = nib.load(os.path.join(preds, file))
        pred = pred_nib.get_fdata()
        voxvol = np.prod(pred_nib.header.get_zooms())
        pred_foreground = (pred > conf).astype(int)
        # check if there is a region over 500mm^3
        # find connected components
        pred_label = label(pred_foreground)
        # get the largest connected component
        pred_label_sizes = np.bincount(pred_label.flat)

        # if empty, positive is false
        if len(pred_label_sizes) == 1:
            positive = False
        else:
            positive = np.sum((pred_label == (np.argmax(pred_label_sizes[1:]) + 1)).astype(int)) * voxvol > 500

        if positive:
            if filename in positives:
                case_wise['all']['tp'] += 1
                if positives[filename] == 'cancer' or positives[filename] == 'uretic_cancer':
                    case_wise['cancers']['tp'] += 1
                    case_wise['solid_masses']['tp'] += 1
                elif positives[filename] == 'mass' or positives[filename] == 'oncocytoma':
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
            if filename in positives:
                case_wise['all']['fn'] += 1
                if positives[filename] == 'cancer' or positives[filename] == 'uretic_cancer':
                    case_wise['cancers']['fn'] += 1
                    case_wise['solid_masses']['fn'] += 1
                elif positives[filename] == 'mass' or positives[filename] == 'oncocytoma':
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

    # plot all curves on the same graph and save
    # also save confidences for each patient
    #save patient confidences as json
    with open(os.path.join(new_fold, 'conf_per_patient.json'), 'w') as f:
        json.dump(per_patient_confidences, f)

    #save results
    np.save(save_loc, results)

    print(results)
    sys.stdout.flush()
    np.save(ppc_path, per_patient_confidences)

    # plot the ROC curves for each group
    plot_results = np.concatenate([results,[[[0,1],[0,1],[0,1]]]],axis=0)

    cancer_auc = np.trapz(plot_results[:,0,1], plot_results[:,0,0])
    mass_auc = np.trapz(plot_results[:,1,1], plot_results[:,1,0])
    all_auc = np.trapz(plot_results[:,2,1], plot_results[:,2,0])

    plt.plot(plot_results[:,0,1], plot_results[:,0,0], label='Cancers (AUC {:.4f})'.format(cancer_auc))
    plt.plot(plot_results[:,1,1], plot_results[:,1,0], label='Solid Masses (AUC {:.4f})'.format(mass_auc))
    plt.plot(plot_results[:,2,1], plot_results[:,2,0], label='All (AUC {:.4f})'.format(all_auc))
    plt.legend()
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curves')
    plt.savefig(png_path)
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