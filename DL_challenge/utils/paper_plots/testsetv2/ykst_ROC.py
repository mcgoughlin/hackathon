#read csv for cancers, all solid tumors, and all potential malignancies
#plot ROC curve for each mass type

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

home = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/ROCs'
save_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper'
subfolders = [folder for folder in os.listdir(home) if os.path.isdir(os.path.join(home, folder))]
cancers_sensspec = []
masses_sensspec = []
alls_sensspec = []
fig = plt.figure(figsize=(8,7))
for folder in subfolders:
    #cancers in cnacers_ROC.csv, all solid tumors in masses_ROC.csv, all potential malignancies in all_ROC.csv
    # open csvs as dataframes, append

    cancers = pd.read_csv(os.path.join(home, folder, 'cancers_ROC.csv'))
    masses = pd.read_csv(os.path.join(home, folder, 'masses_ROC.csv'))
    alls = pd.read_csv(os.path.join(home, folder, 'all_ROC.csv'))

    # append to list of dataframes
    cancers_sens_spec = cancers[['Sensitivity','Specificity']].values
    masses_sens_spec = masses[['Sensitivity','Specificity']].values
    alls_sens_spec = alls[['Sensitivity','Specificity']].values

    cancers_sensspec.append(cancers_sens_spec)
    masses_sensspec.append(masses_sens_spec)
    alls_sensspec.append(alls_sens_spec)

# concat the results into a single array per mass type
cancers_sensspec = np.concatenate(cancers_sensspec)
masses_sensspec = np.concatenate(masses_sensspec)
alls_sensspec = np.concatenate(alls_sensspec)

# drop duplicates in the arrays
cancers_sensspec = np.unique(cancers_sensspec, axis=0)
masses_sensspec = np.unique(masses_sensspec, axis=0)
alls_sensspec = np.unique(alls_sensspec, axis=0)

# sort by sensitivity in ascending order
cancers_sensspec = cancers_sensspec[np.argsort(cancers_sensspec[:,0])]
#then sort by specificity in descending order
cancers_sensspec = cancers_sensspec[np.argsort(cancers_sensspec[:,1])[::-1]]

masses_sensspec = masses_sensspec[np.argsort(masses_sensspec[:,0])]
masses_sensspec = masses_sensspec[np.argsort(masses_sensspec[:,1])[::-1]]

alls_sensspec = alls_sensspec[np.argsort(alls_sensspec[:,0])]
alls_sensspec = alls_sensspec[np.argsort(alls_sensspec[:,1])[::-1]]

#use aestheically pleasing colors for the ROC curves
colors = ['r','b','g']
#plot the ROC curves

for name,data,c in zip(['Cancers\n','All Solid Masses\n'],
                       [cancers_sensspec, masses_sensspec,],
                       colors):
    auc = np.abs(np.trapz(data[:,0], data[:,1]))
    plt.plot(data[:,1]*100, data[:,0]*100, label=name+'(AUC = %.3f)' % auc, color=c, linewidth=2)
plt.xlabel('Specificity / %',fontdict={'fontsize': 16})
plt.ylabel('Sensitivity / %',fontdict={'fontsize': 16})
# add minor and major gridlines
plt.grid(which='both', axis='both')
plt.minorticks_on()
plt.grid(which='minor', axis='both', linestyle='-', linewidth=0.2)

#change tick font size
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)

# find the point in alls_sensspec where sensitivity is 1.0 and specificity is maximised
max_spec = np.max(alls_sensspec[:,1][alls_sensspec[:,0]==1.0])
#drop a vertical line from this point to the x-axis between 0 and 100
plt.plot([max_spec*100,max_spec*100],[-2,100],color='gray',linestyle='--',linewidth=2)
# annotate this vertical line with ' Potential Screening Workload Reduction'
plt.text(max_spec*100 - 4, 50, 'Up to {:.1f}% Screening Workload Reduction'.format(max_spec*100), rotation=90, verticalalignment='center')

# do the same for the cancers_sensspec
max_spec = np.max(cancers_sensspec[:,1][cancers_sensspec[:,0]==1.0])
plt.plot([max_spec*100,max_spec*100],[-2,100],color='gray',linestyle='--',linewidth=2)
# annotate this vertical line with 'High-Risk Cancer Prioritisation'
plt.text(max_spec*100 - 4, 40, 'High-Risk Cancer Prioritisation', rotation=90, verticalalignment='center')
plt.ylim([-2,102])
plt.xlim([-2,102])
plt.grid(which='both')
plt.minorticks_on()
plt.grid(which='minor',axis='both',linestyle='--',linewidth=0.5)
plt.grid(which='major',axis='both',linestyle='-',linewidth=1)

radiologist_cancer_sensspec = [100,100*(4019-(155-13))/4019]
radiologist_mass_sensspec = [100*(24/25),100*(4019-155+25)/4019]
radiologist_all_sensspec = [100*(36/42),100*(4019-155+42)/4019]

# plt.plot(radiologist_cancer_sensspec[1],radiologist_cancer_sensspec[0],'ro',label='Radiologist Cancers',
#             markersize=5)
# plt.plot(radiologist_mass_sensspec[1],radiologist_mass_sensspec[0],'bo',label='Radiologist Masses',
#             markersize=5)
# plt.plot(radiologist_all_sensspec[1],radiologist_all_sensspec[0],'go',label='Radiologist All',
#             markersize=5)
# plt.title('ROC Curves for External Test: YKST',fontdict={'fontsize': 15})
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_loc,'YKST_ROC.png'))
plt.show()

print(cancers_sensspec)
# # loop through the meta dataframes, plot ROC curves
# for df in zip(['Cancers','All Solid Masses','All Potential Malginancies'],[meta_cancers, meta_masses, meta_alls]):
#     tpr = df