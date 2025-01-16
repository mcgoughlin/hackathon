import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Load the data
fp = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/ykst_mass_size.csv'
df = pd.read_csv(fp)

# Discard 'id' and 'Notes', and set data types
df = df.drop(columns=['id', 'Notes'])
df['type'] = df['type'].astype(str)
df['diameter'] = df['diameter'].astype(int)
df['Detected'] = df['Detected'].astype(int)

# Define bin edges for bins like 0-5, 5-10, 10-15, etc.
bin_width = 5
max_diameter = 80  # Set this to the maximum diameter you want to display
bin_edges = np.arange(0, max_diameter + bin_width, bin_width)

# Use pandas.cut to bin 'diameter' into the defined bins
df['diameter_bin'] = pd.cut(df['diameter'], bins=bin_edges, right=False, labels=bin_edges[:-1])

# Convert 'diameter_bin' to numeric type for plotting
df['diameter_bin'] = df['diameter_bin'].astype(float)

# Group and pivot the data
grouped = df.groupby(['diameter_bin', 'type', 'Detected']).size().reset_index(name='count')
pivot_table = grouped.pivot_table(index='diameter_bin', columns=['type', 'Detected'], values='count', fill_value=0)
pivot_table = pivot_table.sort_index(axis=1, level=[0,1])

# Get unique 'type' values and assign colors
types = df['type'].unique()
print(dir(plt.cm))
colors = plt.cm.OrRd(np.linspace(0, 1, len(types)))
type_colors = dict(zip(types, colors))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

diameter_bins = pivot_table.index.values
bar_width = bin_width
bottom = np.zeros(len(diameter_bins))

# Define hatching patterns for 'Detected' status
detected_hatches = {1: '/', 0: ''}

for (type_value, detected_value), counts in pivot_table.items():
    color = type_colors[type_value]
    hatch = detected_hatches[detected_value]
    ax.bar(diameter_bins, counts.values, bar_width, bottom=bottom,
           align='edge', color=color, hatch=hatch, edgecolor='black')
    bottom += counts.values

ax.set_xlabel('Diameter / mm',fontdict={'fontsize': 20})
ax.set_ylabel('Count',fontdict={'fontsize': 20})

# Set x-axis limits and ticks
ax.set_xlim(0, max_diameter)
ax.set_xticks(np.arange(0, max_diameter + bin_width, bin_width))
ax.set_xticklabels(np.arange(0, max_diameter + bin_width, bin_width))

# Create custom legend handles for 'type'
type_handles = [Patch(facecolor=type_colors[type_value], edgecolor='black') for type_value in types]
type_labels = list(types)

# Create custom legend handles for 'Detected' status
hatches = ['/', '']  # Detected == 1: '/', Detected == 0: ''
detected_status = ['Detected', 'Not Detected']
detected_handles = [Patch(facecolor='white', edgecolor='black', hatch=hatch) for hatch in hatches]
detected_labels = detected_status

# Combine both legends into one
labels = ['Type']+ type_labels + ['Detected'] +detected_labels
handles = [Patch(facecolor='none', edgecolor='none',visible=False)] + type_handles + [Patch(facecolor='none', edgecolor='none',visible='False')] + detected_handles

legend = ax.legend(handles=handles, labels=labels, loc='upper right',fontsize='large')

isheader = False
# Make 'Type' and 'Detected' labels bold to act as headers
for text, label in zip(legend.get_texts(), labels):
    print(label)
    if (label == 'Type'):
        print('a')
        text.set_weight('bold')
        text.set_ha('left')  # Align headers to the left
    elif (label == 'Detected') and not isheader:
        print('b')
        text.set_weight('bold')
        text.set_ha('right')
        isheader = True
    else:
        text.set_ha('left')  # Align labels to the left

#set font size in axis ticks and labels
ax.tick_params(axis='both', which='major', labelsize=15)
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')
    label.set_fontsize(15)
plt.tight_layout()
plt.savefig('/Users/mcgoug01/Cambridge University Dropbox/William McGough/YKST/ykst_mass_size.png')
plt.show()


