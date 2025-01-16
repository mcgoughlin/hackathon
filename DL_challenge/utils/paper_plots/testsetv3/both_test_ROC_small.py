import os
import pandas as pd
import numpy as np
import matplotlib.ticker as plticker
import matplotlib.patches as mpatches
import nibabel as nib
seg_loc = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/test_set_v2/labels'

full_dose = False

meta_folder = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions_v3/fulldose_test_ROCv3'
sub_folders = [os.path.join(meta_folder, f,'predictions.csv') for f in os.listdir(meta_folder) if os.path.isdir(os.path.join(meta_folder, f))]
names = [float(f.split('/')[-2]) for f in sub_folders]
save_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper'
# sort sub_folders by names_sub_folders
folders = [x for _,x in sorted(zip(names,sub_folders))]
names_sub_folders = sorted(names)

truth_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internaltest_metadata.csv'
iztok_predfulldose_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/iztokFD.csv'
hania_predfulldose_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/haniaFD.csv'
cathal_predfulldose_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/cathalFD.csv'
folders = [iztok_predfulldose_csv_fp, hania_predfulldose_csv_fp,cathal_predfulldose_csv_fp] + folders
names_sub_folders = ['Iztok','Hania','Cathal'] + names_sub_folders
fulldose_ai_results = [[100,0]]
sizes_dictlist = []

large_cases = []
pile = []
tr = pd.read_csv(truth_csv_fp)
# go through OriginalId in truth_csv_fp, and find the corresponding case in seg_loc
for ogid in tr['OriginalID']:
    studyid = tr[tr['OriginalID']==ogid]['StudyID'].values[0]
    # if case.lower does not begin with kits or rcc, skip
    if not ogid.lower().startswith('kits') and not ogid.lower().startswith('rcc'):
        continue

    # if case begins with kits in seg_loc
    if ogid.lower().startswith('kits'):
        ogid = ogid.lower().replace('kits-','KiTS-')
        seg_n = nib.load(os.path.join(seg_loc,ogid))
        seg = (seg_n.get_fdata()==2).astype(np.uint8)
        seg = np.swapaxes(seg,0,2)
        vox_vol = np.prod(seg_n.header.get_zooms())
        size = np.sum(seg) * vox_vol
        if np.sum(seg)*vox_vol > 33510.32:
            # append to large cases
            large_cases.append(studyid)
        else:
            pile.append(np.sum(seg)*vox_vol)
    # if case begins with rcc in seg_loc
    elif ogid.lower().startswith('rcc'):
        ogid = ogid.lower().replace('rcc_','Rcc_')
        seg_n = nib.load(os.path.join(seg_loc,ogid))
        seg = (seg_n.get_fdata()==1).astype(np.uint8)
        vox_vol = np.prod(seg_n.header.get_zooms())
        size = np.sum(seg) * vox_vol
        if np.sum(seg)*vox_vol > 33510.32:
            # append to large cases
            large_cases.append(studyid)
        else:
            pile.append(np.sum(seg)*vox_vol)
    else:
        size = 0

    sizes_dictlist.append({'studyid':studyid,'originalid':ogid,'size':size})

print('mean small size is',np.mean(pile))

size_df = pd.DataFrame(sizes_dictlist)
size_df.to_csv('/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/internaltest_sizes.csv',index=False)
print(size_df)
print(len(large_cases))

for name,pred_csv_fp in zip(names_sub_folders,folders):
    print(name)

    # pred and columns have 'RightCancer', 'LeftCancer' columns - trust has answers, pred has predictions
    #calc sens and spec
    pred = pd.read_csv(pred_csv_fp)
    truth = pd.read_csv(truth_csv_fp)

    #remove all rows in pred and truth where the StudyID is in large_cases
    pred = pred[~pred['StudyID'].isin(large_cases)]
    truth = truth[~truth['StudyID'].isin(large_cases)]

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
        iztok_full = [sens,spec]
    elif name == 'Cathal':
        cathal_full = [sens,spec]
    elif name == 'Hania':
        hania_full = [sens,spec]
    else:
        if name == 0.120:
            fd_high_sens = [sens, spec]
        elif name == 0.550:
            fd_high_acc = [sens, spec]
        elif name == 0.994:
            fd_high_spec = [sens, spec]
        fulldose_ai_results.append([sens, spec])

fulldose_ai_results.append([0,100])

fulldose_ai_results = np.array(fulldose_ai_results)
# sort by ascending first column and descending second column - merge sort by second column first
fulldose_ai_results = fulldose_ai_results[fulldose_ai_results[:,1].argsort(kind='mergesort')[::-1]]
fulldose_ai_results = fulldose_ai_results[fulldose_ai_results[:,0].argsort(kind='mergesort')]

#calculate AUC for ai_results
fd_auc = np.abs(np.trapz(fulldose_ai_results[:,0],fulldose_ai_results[:,1])/10000)
print(f'AUC: {fd_auc:.3f}')
print(fulldose_ai_results)


lowdose_folder =  '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions_v3/lowdose_test_ROCv3'
truth_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/LD/internaltest_metadata.csv'
ld_sub_folders = [os.path.join(lowdose_folder, f,'predictions.csv') for f in os.listdir(lowdose_folder) if os.path.isdir(os.path.join(lowdose_folder, f))]
names_ld= [float(f.split('/')[-2]) for f in ld_sub_folders]
iztok_predlowdose_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/iztokLD.csv'
cathal_predlowdose_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/cathalLD.csv'
hania_predlowdose_csv_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/haniaLD.csv'
ld_sub_folders = [x for _,x in sorted(zip(names_ld,ld_sub_folders))]
names_ld_folders = sorted(names_ld)

# sort sub_folders by names_sub_folders
ld_sub_folders = [iztok_predlowdose_csv_fp,cathal_predlowdose_csv_fp,hania_predlowdose_csv_fp] + ld_sub_folders
names_ld_folders = ['Iztok','Cathal','Hania'] + names_ld_folders

#repeat for low dose
lowdose_ai_results = [[100,0]]
for name,pred_csv_fp in zip(names_ld_folders,ld_sub_folders):

    # pred and columns have 'RightCancer', 'LeftCancer' columns - trust has answers, pred has predictions
    #calc sens and spec
    pred = pd.read_csv(pred_csv_fp)
    truth = pd.read_csv(truth_csv_fp)

    #remove all rows in pred and truth where the StudyID is in large_cases
    pred = pred[~pred['StudyID'].isin(large_cases)]
    truth = truth[~truth['StudyID'].isin(large_cases)]

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

    if name == 'Iztok' and name != 'Cathal':
        iztok_low = [sens,spec]
    elif name == 'Cathal':
        cathal_low = [sens,spec]
    elif name == 'Hania':
        hania_low = [sens,spec]
    else:
        if name == 0.120:
            ld_high_sens = [sens,spec]
        elif name == 0.550:
            ld_high_acc = [sens,spec]
        elif name == 0.994:
            ld_high_spec = [sens,spec]
        lowdose_ai_results.append([sens,spec])

lowdose_ai_results.append([0,100])
lowdose_ai_results = np.array(lowdose_ai_results)
# sort by ascending first column and descending second column - merge sort by second column first
lowdose_ai_results = lowdose_ai_results[lowdose_ai_results[:,1].argsort(kind='mergesort')[::-1]]
lowdose_ai_results = lowdose_ai_results[lowdose_ai_results[:,0].argsort(kind='mergesort')]

#calculate AUC for ai_results
ld_auc = np.abs(np.trapz(lowdose_ai_results[:,0],lowdose_ai_results[:,1])/10000)
print(f'AUC: {ld_auc:.3f}')

#make marker star
marker = '*'

#plot ROC curve, and plot the point for Iztok and Cathal
# also, plot the operating points for the AI models (high sensitivity, high accuracy, high specificity)
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
fig, ax  = plt.subplots(figsize=(8,7))
fd, = ax.plot(fulldose_ai_results[:,1],fulldose_ai_results[:,0],label='Full-Dose AI\n(AUC = {:.3f})'.format(fd_auc),c='k',zorder=1)
ld, = ax.plot(lowdose_ai_results[:,1],lowdose_ai_results[:,0],label='Low-Dose AI\n(AUC = {:.3f})'.format(ld_auc),c='k',zorder=1,
         linestyle='--')
#plot scatter of fd and ld high sensitivity, high accuracy, high specificity
# plt.scatter([fd_high_sens[1],fd_high_acc[1],fd_high_spec[1]],[fd_high_sens[0],fd_high_acc[0],fd_high_spec[0]],
#             c='k',marker=marker)
# plt.scatter([ld_high_sens[1],ld_high_acc[1],ld_high_spec[1]],[ld_high_sens[0],ld_high_acc[0],ld_high_spec[0]],
#             c='k',marker=marker)
 #plot scatter of trainee in fd and ld
ldtr = ax.scatter([cathal_low[1],hania_low[1]],[cathal_low[0],hania_low[0]],label='Trainee Low',marker='.',s=100,linestyle='--',facecolors='none',edgecolors='b',zorder=2)
ldcr = ax.scatter([iztok_low[1]],[iztok_low[0]],label='Consultant Low',marker='.',s=100,linestyle='--',facecolors='none',edgecolors='r',zorder=2)

# plt.scatter([cathal_full[1],hania_full[1]],[cathal_full[0],hania_full[0]],label='Trainee Radiologists Full Dose',c='r',marker='.')
ax.scatter([iztok_full[1]],[iztok_full[0]],label='Consultant Full',c='r',marker='.',s=100,zorder=2)
ax.scatter([hania_full[1],cathal_full[1]],[hania_full[0],cathal_full[0]],label='Trainee full',c='b',marker='.',s=100,
                zorder=2)
#plot dotted line between dose points
fdcr = ax.plot([iztok_full[1],iztok_low[1]+0.3],[iztok_full[0],iztok_low[0]+0.3],c='r',linestyle=':',zorder=2,linewidth=2)

#plot dotted line for hania
plt.plot([hania_full[1],hania_low[1]+0.3],[hania_full[0],hania_low[0]+0.3],c='b',linestyle=':',zorder=2,linewidth=2)
plt.plot([cathal_full[1],cathal_low[1]-0.1],[cathal_full[0],cathal_low[0]+0.3],c='b',linestyle=':',zorder=2,linewidth=2)
# ax
# .title('ROC Curve: AI Model Detecting RCC in NCCT'.format(fd_auc),fontdict={'fontsize':12})

#annotate previous record
# plt.annotate('Sensitivity 63%\nSpecificity 98%',xy=(prev_record[1],prev_record[0]),xytext=(prev_record[1]-50,prev_record[0]-10),
#              fontsize=16,c='b',
#              arrowprops=dict(color='blue',arrowstyle='->',linewidth=1.5))
#set minor and major grid lines
ax.grid(which='both')
ax.minorticks_on()
ax.grid(which='minor',axis='both',linestyle='--',linewidth=0.5)
ax.grid(which='major',axis='both',linestyle='-',linewidth=1)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
ax.set_xlabel('Specificity / %',fontdict={'fontsize':16,'fontname':'Helvetica'})
ax.set_ylabel('Sensitivity / %',fontdict={'fontsize':16,'fontname':'Helvetica'})
# ax.legend(fontsize=14)

lt = plt.legend(fontsize=16,ncol=1,loc='lower left')
#change lt font to helvetica
for text in lt.get_texts():
    text.set_fontname('Helvetica')
    text.set_fontsize(14)
ax.set_ylim([-2,102])
ax.set_xlim([-2,102])
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'internaltest_ROC_small.png'),dpi=300)
plt.show()

print(fd_high_sens,fd_high_acc,fd_high_spec)
print(ld_high_sens,ld_high_acc,ld_high_spec)

#print all radiologist results
print(iztok_full)
print(hania_full)
print(cathal_full)
print(iztok_low)
print(hania_low)
print(cathal_low)