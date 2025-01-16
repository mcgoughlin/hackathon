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

names = ['Iztok','Cathal','Hania',0.04, 0.7, 0.9999] + names_sub_folders
folders = [iztok_pred_csv_fp, cathal_pred_csv_fp,hania_pred_csv_fp, highsens_pred_csv_fp, highacc_pred_csv_fp, highspec_pred_csv_fp] + sub_folders
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
plt.figure(figsize=(15,15))
plt.switch_backend('TkAgg')
plt.plot(ai_results[:,1],ai_results[:,0],label='AI Model (AUC = {:.3f})'.format(auc),c='k')
plt.scatter(iztok[1],iztok[0],label='Consultant Radiologist',c='r',marker=marker)
plt.scatter([cathal[1],hania[1]],[cathal[0],hania[0]],label='Trainee Radiologists',c='b',marker=marker)
plt.scatter([high_sens[1],high_acc[1],high_spec[1]],[high_sens[0],high_acc[0],high_spec[0]],c=['k'])
plt.scatter([high_acc[1]],[high_acc[0]],c='k',marker=marker)
plt.xlabel('Specificity (%)',fontdict={'fontsize':15})
plt.ylabel('Sensitivity (%)',fontdict={'fontsize':15})
plt.title('ROC Curve on Low Dose Internal Test'.format(auc),fontdict={'fontsize':15})

#write an annotation for the high_acc operating point, with an arrow pointing to it
plt.annotate('Sensitivity 86%\nSpecificity 93%',xy=(high_acc[1],high_acc[0]),xytext=(high_acc[1]-50,high_acc[0]-10),
             fontsize=14,
             arrowprops=dict(color='black',arrowstyle='->',linewidth=1.5))


#plot previous record
prev_record = [63,98]
# plt.scatter(prev_record[1],prev_record[0],label='Radiologist Performance from Literature',c='b',marker=marker)

#annotate previous record
# plt.annotate('Sensitivity 63%\nSpecificity 98%',xy=(prev_record[1],prev_record[0]),xytext=(prev_record[1]-50,prev_record[0]-10),
#              fontsize=16,c='b',
#              arrowprops=dict(color='blue',arrowstyle='->',linewidth=1.5))
#write the model AUC in the bottom left of the plot
# plt.text(0.2,0.8,'AUC = {:.3f}'.format(auc),horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes,
#          fontdict={'color':'black','fontsize':16})

plt.legend(fontsize=15)
#set minor and major grid lines
plt.grid(which='both')
plt.minorticks_on()
plt.grid(which='minor',axis='both',linestyle='--',linewidth=0.5)
plt.grid(which='major',axis='both',linestyle='-',linewidth=1)
plt.xlim([0,105])
plt.ylim([0,105])

plt.show()