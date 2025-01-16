import os
import pandas as pd
import numpy as np

full_dose = False

meta_folder = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/fulldose_ROC'
sub_folders = [os.path.join(meta_folder, f,'predictions.csv') for f in os.listdir(meta_folder) if os.path.isdir(os.path.join(meta_folder, f))]
names = [float(f.split('/')[-2]) for f in sub_folders]

# sort sub_folders by names_sub_folders
folders = [x for _,x in sorted(zip(names,sub_folders))]
names_sub_folders = sorted(names)
truth_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internaltest_metadata.csv'
iztok_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/iztokFD.csv'
hania_pred_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/haniaFD.csv'

folders = [iztok_pred_csv_fp, hania_pred_csv_fp] + folders
names = ['Iztok','Hania'] + names_sub_folders
ai_results = [[100,0]]
for name,pred_csv_fp in zip(names,folders):

    # pred and columns have 'RightCancer', 'LeftCancer' columns - trust has answers, pred has predictions
    #calc sens and spec
    pred = pd.read_csv(pred_csv_fp)
    truth = pd.read_csv(truth_csv_fp)

    # get the truth values
    truth_right = truth['RightCancer'].values
    truth_left = truth['LeftCancer'].values

    # get the predictions
    pred_right = pred['RightCancer'].values
    pred_left = pred['LeftCancer'].values

    # get the number of patients
    num_patients = len(truth_right)

    # get the number of true positives, true negatives, false positives, false negatives
    tp_right = np.sum(np.logical_and(truth_right, pred_right))
    tn_right = np.sum(np.logical_and(np.logical_not(truth_right), np.logical_not(pred_right)))
    fp_right = np.sum(np.logical_and(np.logical_not(truth_right), pred_right))
    fn_right = np.sum(np.logical_and(truth_right, np.logical_not(pred_right)))

    tp_left = np.sum(np.logical_and(truth_left, pred_left))
    tn_left = np.sum(np.logical_and(np.logical_not(truth_left), np.logical_not(pred_left)))
    fp_left = np.sum(np.logical_and(np.logical_not(truth_left), pred_left))
    fn_left = np.sum(np.logical_and(truth_left, np.logical_not(pred_left)))

    #print dataframe rows for all false negatives
    fn_right_indices = np.where(np.logical_and(truth_right, np.logical_not(pred_right)))[0]
    fn_left_indices = np.where(np.logical_and(truth_left, np.logical_not(pred_left)))[0]


    tp = tp_right + tp_left
    tn = tn_right + tn_left
    fp = fp_right + fp_left
    fn = fn_right + fn_left

    sens = 100*tp/(tp+fn)
    spec = 100*tn/(tn+fp)
    print(f'{name}:')
    #print sens and spec to 2dp
    print(f'Sensitivity: {sens:.2f}%')
    print(f'Specificity: {spec:.2f}%')
    print()

    if name == 'Iztok' and name != 'Cathal':
        iztok = [sens,spec]
    elif name == 'Cathal':
        cathal = [sens,spec]
    elif name == 'Hania':
        hania = [sens,spec]
    else:
        if name == 0.04:
            high_sens = [sens,spec]
        elif name == 0.7:
            high_acc = [sens,spec]
        elif name == 0.9999:
            high_spec = [sens,spec]
        ai_results.append([sens,spec])

ai_results.append([0,100])




ai_results = np.array(ai_results)
# sort by ascending first column and descending second column - merge sort by second column first
ai_results = ai_results[ai_results[:,1].argsort(kind='mergesort')[::-1]]
ai_results = ai_results[ai_results[:,0].argsort(kind='mergesort')]


#calculate AUC for ai_results
auc = np.abs(np.trapz(ai_results[:,0],ai_results[:,1])/10000)
print(f'AUC: {auc:.3f}')
print(ai_results)

#make marker star
marker = '*'

#plot ROC curve, and plot the point for Iztok and Cathal
# also, plot the operating points for the AI models (high sensitivity, high accuracy, high specificity)
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
plt.plot(ai_results[:,1],ai_results[:,0],label='AI Model (AUC = {:.3f})'.format(auc),c='k')
plt.scatter(iztok[1],iztok[0],label='Consultant Radiologist',c='r',marker=marker)
plt.scatter(hania[1],hania[0],label='Trainee Radiologist',c='b',marker=marker)
plt.title('ROC Curve on Full Dose Internal Test'.format(auc),fontdict={'fontsize':18})
plt.legend(fontsize=12)
#set minor and major grid lines
plt.grid(which='both')
plt.minorticks_on()
plt.grid(which='minor',axis='both',linestyle='--',linewidth=0.5)
plt.grid(which='major',axis='both',linestyle='-',linewidth=1)
plt.xlim([0,105])
plt.ylim([0,105])

plt.show()