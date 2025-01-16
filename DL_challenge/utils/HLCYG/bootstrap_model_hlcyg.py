import numpy as np
import pandas as pd
import os

samples_per_bootstrap = 147
number_samples = 10000
high_spec = 0.90
low_spec = 0.5
take_average = False
small_ratio = 19/43
# https://link.springer.com/article/10.1007/s00261-017-1376-0

home_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/HLCYG/detection_rocs/'
models = [folder for folder in os.listdir(home_loc) if os.path.isdir(os.path.join(home_loc, folder))]
#filter out model names that do not contain 'noiseonly'
metaresults = []
for model in models:
    print(model)
    conf_per_patient_fp = os.path.join(home_loc,model,'conf_per_patient.npy')
    conf_intervals_fp = os.path.join(home_loc,model,'conf_ROC.csv')
    sizes_fp = os.path.join(home_loc,model,'sizes.csv')
    if not os.path.exists(sizes_fp):
        continue
    large = pd.read_csv(sizes_fp)['Size'].values > ((15**3)*np.pi*4/3)
    noise = pd.read_csv(sizes_fp)['File'].str.contains('noise').values
    conf_per_patient = np.load(conf_per_patient_fp)
    conf_per_patient = conf_per_patient[noise]

    conf_intervals = pd.read_csv(conf_intervals_fp)['Confidence'].values
    num_conf_intervals = len(conf_per_patient[0,:,0])

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

    # interpolate ROC curve and standard deviations to fit between 1000 points between 0 and 1
    interp = np.linspace(0,1,1000)
    interp_sens = np.interp(interp, spec, sens)
    interp_conf = np.interp(interp, spec, conf_intervals)
    interp_sens_std = np.interp(interp, spec, sens_std)
    interp_spec_std = np.interp(interp, spec, spec_std)

    #extract the 98% specificity operating point
    index = np.argmin(np.abs(interp-high_spec))
    highspec_sens = interp_sens[index]
    dose = float(model.split('_')[-1])
    noise_only= 'only' in model
    std = interp_sens_std[index]
    entry = {'Dose':dose,'NoiseOnly':noise_only, 'All Cancers':AUC, 'Sensitivity':highspec_sens, 'SensStdDev':std}

    #redo above but for small_ratio = 1
    small_results = []
    for conf in range(num_conf_intervals):
        conf_res = []
        for sample in range(number_samples):
            relevant_interval = conf_per_patient[:,conf,:]
            small_tumours = relevant_interval[~large]
            small_indices = np.random.randint(0,len(small_tumours), size=int(samples_per_bootstrap*small_ratio))
            sample = small_tumours[small_indices]


            tp, tn, fp, fn = sample.sum(axis=0)
            sens = tp/(tp+fn)
            spec = (tn/(tn+fp))
            conf_res.append((sens, spec))
        conf_res = np.array(conf_res)
        small_results.append(conf_res)

    small_results = np.array(small_results)
    means_small = small_results.mean(axis=1)
    stds_small = small_results.std(axis=1)

    # plot an ROC with means, with error bars of 1 std
    sens_small = means_small[:,0]
    spec_small = means_small[:,1]
    sens_std_small = stds_small[:,0]
    spec_std_small = stds_small[:,1]

    # prepend and append 0 to sens and spec
    sens_small = np.concatenate(([1],sens_small,[0]))
    spec_small = np.concatenate(([0],spec_small,[1]))

    #prepend and append 0 to sens_std and spec_std
    sens_std_small = np.concatenate(([0],sens_std_small,[0]))
    spec_std_small = np.concatenate(([0],spec_std_small,[0]))

    #find AUC
    AUC_small = np.trapz(sens_small, spec_small)

    # interpolate ROC curve and standard deviations to fit between 1000 points between 0 and 1
    interp_small = np.linspace(0,1,1000)
    interp_sens_small = np.interp(interp_small, spec_small, sens_small)
    interp_conf_small = np.interp(interp_small, spec_small, conf_intervals)
    interp_sens_std_small = np.interp(interp_small, spec_small, sens_std_small)
    interp_spec_std_small = np.interp(interp_small, spec_small, spec_std_small)

    #extract the 98% specificity operating point
    index_small = np.argmin(np.abs(interp_small-high_spec))
    highspec_sens_small = interp_sens_small[index_small]
    std_small = interp_sens_std_small[index_small]
    entry_small = {'Small Cancers':AUC_small, 'Sensitivity_small':highspec_sens_small, 'SensStdDev_small':std_small}

    #merge the two entries
    entry = {**entry, **entry_small}
    print(entry)
    metaresults.append(entry)
print(metaresults)
df = pd.DataFrame(metaresults)
df = df.sort_values('All Cancers')
print(df)

noise_analysis = pd.read_csv('/Users/mcgoug01/Cambridge University Dropbox/William McGough/HLCYG/noise_analysis.csv')
#take the column-wise average of all columns in noise_analysis except 'file'
noise_analysis = noise_analysis.drop(columns='file')
noise_mean = noise_analysis.mean(axis=0).values[1:]
noise_std = noise_analysis.std(axis=0).values[1:]
#each column is a dose, each row is the mean noise. Extract these two sets of values
doses = noise_analysis.columns.values[1:]

#wherevever the dose is in the df, add the mean noise and std noise
df['MeanNoise'] = np.nan
df['StdNoise'] = np.nan
for i,dose in enumerate(doses):
    if float(dose) in df['Dose'].values:
        df.loc[df['Dose'] == float(dose), 'MeanNoise'] = noise_mean[i]
        df.loc[df['Dose'] == float(dose), 'StdNoise'] = noise_std[i]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Assuming 'df' is your DataFrame with the proper columns
df['Training Data'] = df['NoiseOnly'].map({False: 'LDCT + NCCT', True: 'LDCT Only'})

# Melt the DataFrame for plotting
df_melted = df.melt(
    id_vars=['MeanNoise', 'Training Data'],
    value_vars=['All Cancers', 'Small Cancers'],
    var_name='Cancer Type',
    value_name='AUC_Value'
)

df_melted.to_csv('/Users/mcgoug01/Cambridge University Dropbox/William McGough/HLCYG/results_df.csv')

# Define a more sophisticated style
sns.set(style="ticks", context="talk")
plt.style.use(["seaborn-poster", "seaborn-ticks"])

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# Define refined markers and palette
markers = {'All Cancers': '^', 'Small Cancers': 'o'}
palette = {'LDCT + NCCT': '#0072B2', 'LDCT Only': '#D55E00'}

# Plot using seaborn scatterplot
scatter = sns.scatterplot(
    data=df_melted,
    x='MeanNoise',
    y='AUC_Value',
    hue='Training Data',
    style='Cancer Type',
    markers=markers,
    palette=palette,
    s=100,
    alpha=0.8,
    ax=ax
)

# Calculate and plot regression line and R^2 value for each group
for name, group in df_melted.groupby(['Training Data', 'Cancer Type']):
    # Fit the regression model
    model = LinearRegression()
    X = group['MeanNoise'].values.reshape(-1, 1)
    y = group['AUC_Value']
    model.fit(X, y)

    # Define x range for regression line plotting
    x_range = np.linspace(X.min(), X.max(), 100)
    predictions = model.predict(x_range.reshape(-1, 1))

    # Plot regression line
    if 'All Cancers' in name:
        ax.plot(x_range, predictions, color=palette[name[0]], linestyle='-', alpha=0.5)
    else:
        ax.plot(x_range, predictions, color=palette[name[0]], linestyle='--',alpha=0.5)

# Improve axis labels and title
ax.set_xlabel('Mean Noise', fontsize=16, weight='bold')
ax.set_ylabel('AUC Value', fontsize=16, weight='bold')
ax.set_title('Dose vs. AUC Analysis', fontsize=18, weight='bold', pad=20)

# Customize the legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles=handles, labels=labels, title='Data and AUC Types', loc='upper left', bbox_to_anchor=(1, 1))

# Fine-tune grid and axes appearance
sns.despine(trim=True)
ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
ax.set_axisbelow(True)

# Adjust layout to prevent clipping
plt.tight_layout()

# Optionally save the figure with high resolution
# plt.savefig('enhanced_dose_vs_auc_scatter_plot.png', dpi=300)

# Display the plot
plt.show()