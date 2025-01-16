import numpy as np
import pandas as pd
import os


sizes_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/internaltest_sizes.csv'
ld_conf_per_patient_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/lowdose_ROC/percase_confidences.csv'
fd_conf_per_patient_fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/fulldose_ROC/percase_confidences.csv'
labels ='/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/FD/internaltest_metadata.csv'
labels_df = pd.read_csv(labels)

# load confidences
ld_conf_per_patient = pd.read_csv(ld_conf_per_patient_fp)[['RightCancer', 'LeftCancer','RLabel', 'LLabel']]
ld_conf_per_patient['RLabel'] = ld_conf_per_patient['RLabel']>0.5
ld_conf_per_patient['LLabel'] = ld_conf_per_patient['LLabel']>0.5

fd_conf_per_patient = pd.read_csv(fd_conf_per_patient_fp)[['RightCancer', 'LeftCancer','RLabel', 'LLabel']]
fd_conf_per_patient['RLabel'] = fd_conf_per_patient['RLabel']>0.5
fd_conf_per_patient['LLabel'] = fd_conf_per_patient['LLabel']>0.5

sizes = pd.read_csv(sizes_fp)
ld_conf_per_patient['size'] = 0
fd_conf_per_patient['size'] = 0
#where studyid index matches the index in the confidences, we can add the size to the confidences,

for studyid in sizes['studyid']:
    index = int(studyid.replace('ID', '')) - 1
    ld_conf_per_patient.loc[index, 'size'] = sizes[studyid==sizes['studyid']]['size'].values[0]
    fd_conf_per_patient.loc[index, 'size'] = sizes[studyid==sizes['studyid']]['size'].values[0]

ld_conf_per_patient['large'] = ld_conf_per_patient['size'] > ((15**3)*np.pi*4/3)
fd_conf_per_patient['large'] = fd_conf_per_patient['size'] > ((15**3)*np.pi*4/3)

#only consider cases where a patient has a cancer (i.e., where size>100)
ld_conf_per_patient = ld_conf_per_patient[ld_conf_per_patient['size']>100]
fd_conf_per_patient = fd_conf_per_patient[fd_conf_per_patient['size']>100]
print(ld_conf_per_patient.head(59))
#print number of small and large tumours
print(np.sum(ld_conf_per_patient['large']))
print(np.sum(~ld_conf_per_patient['large']))
print(len(ld_conf_per_patient))

confidences = np.linspace(0,1,101)
samples_per_bootstrap = 59
number_samples = 10000
small_ratio = 19/43
small_sample_size = int(samples_per_bootstrap * small_ratio)
large_sample_size = samples_per_bootstrap - small_sample_size

print(small_sample_size, large_sample_size)
# https://link.springer.com/article/10.1007/s00261-017-1376-0

ld_results = []
fd_results = []
for conf in confidences:
    ld_conf_res = []
    fd_conf_res = []
    print(conf)
    for sample in range(number_samples):
        # in ld_conf_per_patient and fd_conf_per_patient, we have the confidences for each patient
        # we want to sample with replacement from these confidences, if it exceeds the threshold, we count it as a positive
        # we then calculate the sensitivity and specificity of this sample.

        #we sample at a rate according to the ratio of large to small tumours


        ld_large_sample = ld_conf_per_patient[ld_conf_per_patient['large']].sample(large_sample_size, replace=True)
        ld_small_sample = ld_conf_per_patient[~ld_conf_per_patient['large']].sample(small_sample_size, replace=True)

        fd_large_sample = fd_conf_per_patient[fd_conf_per_patient['large']].sample(large_sample_size, replace=True)
        fd_small_sample = fd_conf_per_patient[~fd_conf_per_patient['large']].sample(small_sample_size, replace=True)

        ld_sample = pd.concat([ld_large_sample, ld_small_sample])
        fd_sample = pd.concat([fd_large_sample, fd_small_sample])

        ld_sample['RPositive'] = ld_sample['RightCancer'] > conf
        ld_sample['LPositive'] = ld_sample['LeftCancer'] > conf
        fd_sample['RPositive'] = fd_sample['RightCancer'] > conf
        fd_sample['LPositive'] = fd_sample['LeftCancer'] > conf

        # we then calculate the sensitivity and specificity of this sample.
        ld_tp = np.sum(ld_sample['RPositive'] & fd_sample['RLabel']) + np.sum(ld_sample['LPositive'] & fd_sample['LLabel'])
        ld_tn = np.sum(~ld_sample['RPositive'] & ~fd_sample['RLabel']) + np.sum(~ld_sample['LPositive'] & ~fd_sample['LLabel'])
        ld_fp = np.sum(ld_sample['RPositive'] & ~fd_sample['RLabel']) + np.sum(ld_sample['LPositive'] & ~fd_sample['LLabel'])
        ld_fn = np.sum(~ld_sample['RPositive'] & fd_sample['RLabel']) + np.sum(~ld_sample['LPositive'] & fd_sample['LLabel'])

        fd_tp = np.sum(fd_sample['RPositive'] & ld_sample['RLabel']) + np.sum(fd_sample['LPositive'] & ld_sample['LLabel'])
        fd_tn = np.sum(~fd_sample['RPositive'] & ~ld_sample['RLabel']) + np.sum(~fd_sample['LPositive'] & ~ld_sample['LLabel'])
        fd_fp = np.sum(fd_sample['RPositive'] & ~ld_sample['RLabel']) + np.sum(fd_sample['LPositive'] & ~ld_sample['LLabel'])
        fd_fn = np.sum(~fd_sample['RPositive'] & ld_sample['RLabel']) + np.sum(~fd_sample['LPositive'] & ld_sample['LLabel'])

        ld_sens = ld_tp/(ld_tp+ld_fn)
        ld_spec = ld_tn/(ld_tn+ld_fp)
        ld_conf_res.append([ld_sens, ld_spec])

        fd_sens = fd_tp/(fd_tp+fd_fn)
        fd_spec = fd_tn/(fd_tn+fd_fp)
        fd_conf_res.append([fd_tp/(fd_tp+fd_fn), fd_tn/(fd_tn+fd_fp)])
    ld_results.append(ld_conf_res)
    fd_results.append(fd_conf_res)
    print(np.mean(ld_conf_res, axis=0))
    print(np.mean(fd_conf_res, axis=0))

ld_results = np.array(ld_results)
fd_results = np.array(fd_results)

# take means and std deviations at each confidence level
ld_means = np.mean(ld_results, axis=1)
ld_stds = np.std(ld_results, axis=1)
fd_means = np.mean(fd_results, axis=1)
fd_stds = np.std(fd_results, axis=1)

#take AUCs of the ROC curves
ld_aucs = np.trapz(ld_means[:,1], ld_means[:,0])
fd_aucs = np.trapz(fd_means[:,1], fd_means[:,0])

print('Low dose AUC: ', ld_aucs)
print('Full dose AUC: ', fd_aucs)

#plot this in an aesthetically pleasing manner. black lines for both curves, dashed for low-dose
import matplotlib.pyplot as plt
plt.plot(fd_means[:,0], fd_means[:,1], color='black', label='Full dose')
plt.plot(ld_means[:,0], ld_means[:,1], color='black', linestyle='dashed', label='Low dose')
# plt.fill_between(fd_means[:,0], fd_means[:,1]-fd_stds[:,1], fd_means[:,1]+fd_stds[:,1], color='black', alpha=0.2)
# plt.fill_between(ld_means[:,0], ld_means[:,1]-ld_stds[:,1], ld_means[:,1]+ld_stds[:,1], color='black', alpha=0.2)
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.legend()
plt.savefig('/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/bootstrap_internal.png')
plt.show()

#save the results
results = pd.DataFrame({'confidences': confidences, 'ld_sens': ld_means[:,0], 'ld_spec': ld_means[:,1], 'fd_sens': fd_means[:,0], 'fd_spec': fd_means[:,1],
                        'ld_sens_std': ld_stds[:,0], 'ld_spec_std': ld_stds[:,1], 'fd_sens_std': fd_stds[:,0], 'fd_spec_std': fd_stds[:,1]})
results.to_csv('/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/bootstrap_results_internal.csv', index=False)

