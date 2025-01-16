import numpy as np
import pandas as pd
import os

sizes_fp = '/Users/mcgoug01/Downloads/bootstrap/sizes.csv'
conf_per_patient_fp = '/Users/mcgoug01/Downloads/bootstrap/conf_per_patient.npy'
conf_intervals_fp = '/Users/mcgoug01/Downloads/bootstrap/conf_ROC.csv'
large = pd.read_csv(sizes_fp)['size'].values > ((15**3)*np.pi*4/3)
conf_per_patient = np.load(conf_per_patient_fp)
conf_intervals = pd.read_csv(conf_intervals_fp)['Confidence'].values

samples_per_bootstrap = 294
number_samples = 10000
num_conf_intervals = len(conf_per_patient[0,:,0])
high_spec = 0.98
low_spec = 0.5
take_average = False

small_ratio = 19/43
# small_ratio = 1
# https://link.springer.com/article/10.1007/s00261-017-1376-0

results = []
for conf in range(num_conf_intervals):
    conf_res = []
    for sample in range(number_samples):
        relevant_interval = conf_per_patient[:,conf,:]

        small_tumours = relevant_interval[~large]
        large_tumours = relevant_interval[large]

        small_indices = np.random.randint(0,len(small_tumours), size=int(samples_per_bootstrap*small_ratio))
        large_indices = np.random.randint(0,len(large_tumours), size=int(samples_per_bootstrap*(1-small_ratio)))

        small_sample = small_tumours[small_indices]
        large_sample = large_tumours[large_indices]

        sample = np.concatenate((small_sample, large_sample))

        tp, tn, fp, fn = sample.sum(axis=0)
        sens = tp/(tp+fn)
        spec = (tn/(tn+fp))
        conf_res.append((sens, spec))
    conf_res = np.array(conf_res)
    results.append(conf_res)

results = np.array(results)
means = results.mean(axis=1)
stds = results.std(axis=1)

# plot an ROC with means, with error bars of 1 std
sens = means[:,0]
spec = means[:,1]
sens_std = stds[:,0]
spec_std = stds[:,1]

# prepend and append 0 to sens and spec
sens = np.concatenate(([1],sens,[0]))
spec = np.concatenate(([0],spec,[1]))

#prepend and append 0 to sens_std and spec_std
sens_std = np.concatenate(([0],sens_std,[0]))
spec_std = np.concatenate(([0],spec_std,[0]))

#find AUC
AUC = np.trapz(sens, spec)

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
fig = plt.figure(figsize=(6,6))
plt.plot(1-spec, sens, c = 'b', label='ROC curve')
#plot semi-transparent error line with fill between
# plt.fill_between(1-spec, sens-1.96*sens_std, sens+1.96*sens_std, alpha=0.2, color='b', label='95% Confidence Interval')
plt.xlabel('1-Specificity', fontsize=14)
plt.ylabel('Sensitivity', fontsize=14)
#plot minor and major grid lines
plt.grid(which='both')
plt.minorticks_on()
#make minor gridlines more faint and appear every 0.05
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)

#plot diagonal line
plt.plot([0,1],[0,1], 'k--')



# interpolate ROC curve and standard deviations to fit between 1000 points between 0 and 1
interp = np.linspace(0,1,1000)
interp_sens = np.interp(interp, spec, sens)
interp_conf = np.interp(interp, spec, conf_intervals)
interp_sens_std = np.interp(interp, spec, sens_std)
interp_spec_std = np.interp(interp, spec, spec_std)

#extract the 98% specificity operating point
index = np.argmin(np.abs(interp-high_spec))
#find the index of the +1.96 standard deviation point and -1.96 standard deviation point
index_plus = np.argmin(np.abs(interp-high_spec+1.96*interp_spec_std[index]))
index_minus = np.argmin(np.abs(interp-high_spec-1.96*interp_spec_std[index]))
#print the confidence interval at this point

highspec_sens = interp_sens[index]
highspec_conf = interp_conf[index]
highspec_spec = interp[index]

# print everything we know about this value
print('Index of 98% specificity: {}'.format(index))
print('Sensitivity at 98% specificity: {:.5f} +- {:.5f}'.format(interp_sens[index], interp_sens_std[index]*1.96))
print('Confidence at 98% specificity: {:.5f}+-{:.5f}'.format(interp_conf[index], np.abs(interp_conf[index_plus]-interp_conf[index_minus])/2))
print('Specificity at 98% specificity: {:.5f}=-{:.5f}'.format(interp[index], np.abs(interp[index_plus]-interp[index_minus])/2))
print()

# do the same but for 95% sensitivity
index = np.argmin(np.abs(interp-low_spec))
index_plus = np.argmin(np.abs(interp-low_spec+1.96*interp_sens_std[index]))
index_minus = np.argmin(np.abs(interp-low_spec-1.96*interp_sens_std[index]))
print('Index of 95% sensitivity: {}'.format(index))
print('Specificity at 95% sensitivity: {:.5f}+-{:.5f}'.format(interp[index], np.abs(interp[index_plus]-interp[index_minus])/2))
print('Sensitive at 95% sensitivity: {:.5f}+-{:.5f}'.format(interp_sens[index], interp_sens_std[index]*1.96))
print('Confidence at 95% sensitivity: {:.5f}+-{:.5f}'.format(interp_conf[index], np.abs(interp_conf[index_plus]-interp_conf[index_minus])/2))
print()

highsens_sens = interp_sens[index]
highsens_spec = interp[index]
highsens_conf = interp_conf[index]





if take_average:
    # make confidence the average of the two previous two
    conf = (interp_conf[index] + interp_conf[np.argmin(np.abs(interp-high_spec))])/2
    # find index of this point
    print(conf)
    index = np.argmin(np.abs(interp_conf-conf))
    index_plus = np.argmin(np.abs(interp_conf-conf+1.96*interp_sens_std[index]))
    index_minus = np.argmin(np.abs(interp_conf-conf-1.96*interp_sens_std[index]))
else:
    # find highest average of sensitivity and specificity
    index = np.argmax(interp_sens+interp)
    index_plus = np.argmin(np.abs(interp[index]+1.96*interp_sens_std[index]))
    index_minus = np.argmin(np.abs(interp[index]-1.96*interp_sens_std[index]))

print('Index of furthest point from diagonal: {}'.format(index))
print('Specificity at furthest point from diagonal: {:.5f}+-{:.5f}'.format(interp[index], np.abs(interp[index_plus]-interp[index_minus])/2))
print('Sensitivity at furthest point from diagonal: {:.5f}+-{:.5f}'.format(interp_sens[index], interp_sens_std[index]*1.96))
print('Confidence at furthest point from diagonal: {:.5f}+-{:.5f}'.format(interp_conf[index], np.abs(interp_conf[index_plus]-interp_conf[index_minus])/2))

highdist_sens = interp_sens[index]
highdist_spec = interp[index]
highdist_conf = interp_conf[index]

# plot these operating points on the curve
# plt.plot([1-highspec_spec,1-highdist_spec, 1-highsens_spec], [highspec_sens, highdist_sens, highsens_sens], 'ro',
#             label='Operating Points')

#plot legend# set tick label size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
print(AUC)
#save

plt.title('Receiver Operating Characteristic, AUC = {:.2f}'.format(AUC),fontdict={'fontsize':16})
plt.tight_layout()

#highlight the area under the curve
plt.fill_between(1-spec, sens, alpha=0.2, color='b', label='AUC')
plt.savefig('ROC_example.png')

# include op point confidences to 4 dp in titles

plt.show()
plt.close()
