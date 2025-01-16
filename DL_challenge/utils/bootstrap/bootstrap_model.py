import numpy as np
import pandas as pd
import os

sizes_fp = '/Users/mcgoug01/Downloads/valid_v3/sizes_cect.csv'
conf_per_patient_fp = '/Users/mcgoug01/Downloads/valid_v3/results_confwise_cect.npy'
sizes = np.genfromtxt(sizes_fp, delimiter=',')
large = sizes > ((15**3)*np.pi*4/3)
conf_per_patient = np.load(conf_per_patient_fp)

confidence_thresholds = np.append(np.arange(0,0.1,0.01),np.arange(0.1, 0.9, 0.05))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.9,0.98,0.02))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.98,1.005,0.005))
conf_intervals = confidence_thresholds

samples_per_bootstrap = len(sizes)
number_samples = 10000
num_conf_intervals = len(conf_intervals)
high_spec = 0.98
low_spec = 0.5**0.5

small_ratio = 19/43
# small_ratio = 1
# https://link.springer.com/article/10.1007/s00261-017-1376-0

results = []
nonbootstrapped_results = []
for conf in range(num_conf_intervals):
    conf_res = []
    for sample in range(number_samples):
        relevant_interval = conf_per_patient[conf,:]

        small_tumours = relevant_interval[~large]
        large_tumours = relevant_interval[large]

        small_indices = np.random.randint(0,len(small_tumours), size=int(samples_per_bootstrap*small_ratio))
        large_indices = np.random.randint(0,len(large_tumours), size=int(samples_per_bootstrap*(1-small_ratio)))

        small_sample = small_tumours[small_indices]
        large_sample = large_tumours[large_indices]

        sample = np.concatenate((small_sample, large_sample))

        tp, fp, fn, tn = sample.sum(axis=0)
        sens = tp/(tp+fn)
        spec = (tn/(tn+fp))
        conf_res.append((sens, spec))
    conf_res = np.array(conf_res)
    results.append(conf_res)
    # calculate non-bootstrapped results from the mean of the conf_per_patient
    tp, fp, fn, tn = conf_per_patient[conf,:].sum(axis=0)
    nonbs_sens = tp/(tp+fn)
    nonbs_spec = (tn/(tn+fp))
    nonbootstrapped_results.append((nonbs_sens, nonbs_spec))

results = np.array(results)
means = results.mean(axis=1)
stds = results.std(axis=1)

nonbootstrapped_results = np.array(nonbootstrapped_results)

# plot an ROC with means, with error bars of 1 std
sens = means[:,0]
spec = means[:,1]
sens_std = stds[:,0]
spec_std = stds[:,1]

# prepend and append 0 to sens and spec
sens = np.concatenate(([1],sens,[0]))
spec = np.concatenate(([0],spec,[1]))

conf_intervals = np.array(conf_intervals)
conf_intervals = np.concatenate(([0],conf_intervals,[1]))

#prepend and append 0 to sens_std and spec_std
sens_std = np.concatenate(([0],sens_std,[0]))
spec_std = np.concatenate(([0],spec_std,[0]))

#find AUC
AUC = np.trapz(sens, spec)

import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.switch_backend('TkAgg')
# subplot of two figures in one row - one with width 7, other with width 3, makign a cumulative
# width of 10. both with height 8
fig = plt.figure(figsize=(12,8))


# create grid for different subplots
gridspec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[8, 3], wspace=0.2,
                         )

ax = fig.add_subplot(gridspec[0])

#plot the non-bootstrapped ROC curve from nonbootstrapped_results
nonbs_sens, nonbs_spec = nonbootstrapped_results.T
nonbs_sens = np.concatenate(([1],nonbs_sens,[0]))
nonbs_spec = np.concatenate(([0],nonbs_spec,[1]))
ax.plot(nonbs_spec*100, nonbs_sens*100, c = 'blue', label='Non-bootstrapped ROC curve (AUC {:.3f})'.format(np.trapz(nonbs_sens, nonbs_spec)), linewidth=3,
        linestyle=':')

ax.plot(spec*100, sens*100, c = 'black', label='Bootstrapped ROC curve (AUC {:.3f})'.format(AUC), linewidth=3)
#plot semi-transparent error line with fill between
ax.fill_between(spec*100, (sens-1.96*sens_std)*100, (sens+1.96*sens_std)*100, alpha=0.2, color='black', label='Bootstrapped 95% confidence interval')
ax.set_xlabel('Specificity / %', fontsize=18, fontname='Helvetica')
ax.set_ylabel('Sensitivity / %', fontsize=18, fontname='Helvetica')
#plot minor and major grid lines
ax.grid(which='both')
ax.minorticks_on()
#make minor gridlines more faint and appear every 0.05
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

#increase tick font size
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

#xlim and ylim -2 to 102
ax.set_xlim(-2,102)
ax.set_ylim(-2,102)

#plot diagonal line
ax.plot([100,0],[0,100], 'k--')

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
ax.plot(np.array([highspec_spec,highdist_spec, highsens_spec])*100,
         np.array([highspec_sens, highdist_sens, highsens_sens])*100,
         'ro', label='Selected operating points')
ax.grid(which='both')
ax.minorticks_on()
ax.grid(which='minor',axis='both',linestyle='--',linewidth=0.5)
ax.grid(which='major',axis='both',linestyle='-',linewidth=1)

#plot legend
lt = ax.legend(loc='lower left', fontsize=14)
#change lt font to helvetica
for text in lt.get_texts():
    text.set_fontname('Helvetica')
    text.set_fontsize(14)
print(AUC)
#save
# plt.savefig('/Users/mcgoug01/Downloads/bootstrap/ROC.png')

# include op point confidences to 4 dp in titles
# ax.title('Receiver Operating Characteristic, AUC = {:.4f}'.format(AUC), fontsize=20)

# in second plot, the ratio of large to small tumours
fraction_large = np.sum(large)/len(large)
fraction_small = 1-fraction_large

ax1 = fig.add_subplot(gridspec[1])
#diffentiate bars by texturing and pastel colours
ax1.bar(['Large', 'Small'], [fraction_large*100, fraction_small*100],color=['palegoldenrod', 'lightsteelblue'], hatch=['\\', '/'],
        edgecolor='black', linewidth=2)
#place text of percentages on top of bars
for i, v in enumerate([fraction_large*100, fraction_small*100]):
    ax1.text(i, v + 1, str(round(v, 1))+'%', color='black', ha='center', fontsize=16,
                fontname='Helvetica')

ax1.set_ylabel('Percentage of tumours in ' +r"$\mathbf{Finetune}$"+" dataset", fontsize=16,
                fontname='Helvetica')
ax1.set_xlabel('Tumour size', fontsize=16, fontname='Helvetica')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=14)
plt.tight_layout()
#change width of ax[1] to 3
plt.savefig('/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper/cvROCv3cect.png')
plt.show()
