import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df_melted = pd.read_csv('/Users/mcgoug01/Cambridge University Dropbox/William McGough/HLCYG/results_df.csv')

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
ax.set_xlabel('Mean Noise / HU', fontsize=16, weight='bold')
ax.set_ylabel('AUC Value', fontsize=16, weight='bold')
ax.set_title('Dose vs. AUC Analysis', fontsize=18, weight='bold', pad=20)
# Customize the legend
# handles, labels = ax.get_legend_handles_labels()
# legend = ax.legend(handles=handles, labels=labels, title='Data and AUC Types', loc='upper left', bbox_to_anchor=(1, 1))
#customize the legend and add the regression lines
handles, labels = ax.get_legend_handles_labels()

#remove the last two handles and labels
handles = handles[:-2]
labels = labels[:-2]

#add the regression lines to the legend
handles.append(plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.5,marker='^', markersize=10))
labels.append('All Cancers')
handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,marker='o', markersize=10))
labels.append('Small Cancers')
#add an empty handle for the legend


legend = ax.legend(handles=handles, labels=labels, title='Data and AUC Types', loc='upper left', bbox_to_anchor=(1, 1))
# Add a legend title
legend.get_title().set_fontsize('16')
# Fine-tune grid and axes appearance
sns.despine(trim=True)
ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
ax.set_axisbelow(True)

# Adjust layout to prevent clipping
plt.tight_layout()

# Optionally save the figure with high resolution
plt.savefig('/Users/mcgoug01/Cambridge University Dropbox/William McGough/HLCYG/hlcyg_cv.png', dpi=300)

# Display the plot
plt.show()