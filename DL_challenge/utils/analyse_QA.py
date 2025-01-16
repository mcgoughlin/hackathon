import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import nibabel as nib
plt.switch_backend('TkAgg')

home = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/QA'
save_loc = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper'
hania_response = pd.read_csv(home + '/hania_qa.csv')
iztok_response = pd.read_csv(home + '/iztok_qa.csv')
cathal_response = pd.read_csv(home + '/cathal_qa.csv')
img_loc = os.path.join(home + '/images')
# loop through all images, extract StudyID and z_spacing
im_zspacing = []
for img in os.listdir(img_loc):
    if not img.endswith('.nii.gz'):
        continue
    im = nib.load(os.path.join(img_loc, img))
    print(im.header['pixdim'])
    im_zspacing.append([img, im.header['pixdim'][3]])
    print([img, im.header['pixdim'][3]])

# create dataframe of image names and z_spacing
im_zspacing = pd.DataFrame(im_zspacing, columns=['StudyID', 'z_spacing'])
im_zspacing['StudyID'] = im_zspacing['StudyID'].astype(str).str[7:-7]
print(im_zspacing)

hania_response['StudyID'] = hania_response['StudyID'].astype(str).str[7:].astype(int)
iztok_response['StudyID'] = iztok_response['StudyID'].astype(str).str[7:].astype(int)
cathal_response['StudyID'] = cathal_response['StudyID'].astype(str).str[7:].astype(int)

hania_response['What type of image is this?'] = hania_response['What type of image is this?'].astype(str)
iztok_response['What type of image is this?'] = iztok_response['What type of image is this?'].astype(str)
cathal_response['What type of image is this?'] = cathal_response['What type of image is this?'].astype(str)

hania_response['Quality? (1-5)'] = hania_response['Quality? (1-5)'].astype(float)
cathal_response['Quality? (1-5)'] = cathal_response['Quality? (1-5)'].astype(float)
iztok_response['Quality? (1-5)'] = iztok_response['Quality? (1-5)'].astype(float)
stddev = pd.read_csv('/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/Bill/QA/std_qa.csv')
stddev['StudyID'] = stddev['StudyID'].astype(str).str[7:]

labels = pd.read_csv(home + '/metadata.csv')
#change labels 'study_id' to 'StudyID'
labels = labels.rename(columns={'study_id': 'StudyID'})
labels['StudyID'] = labels['StudyID'].astype(int)

# Merge the two dataframes
df = pd.merge(hania_response, labels, on='StudyID', how='outer')
df = df.rename(columns={'original_filepath': 'label',
                        'What type of image is this?':'pred_hania',
                        'Quality? (1-5)':'quality_hania'}).astype(str)
df['StudyID'] = df['StudyID'].astype(int)
print(df['StudyID'])
print(iztok_response['StudyID'])
df = pd.merge(iztok_response, df, on='StudyID', how='outer')
df = df.rename(columns={'What type of image is this?':'pred_iztok',
                        'Quality? (1-5)':'quality_iztok'}).astype(str)

df['StudyID'] = df['StudyID'].astype(int)
df = pd.merge(cathal_response, df, on='StudyID', how='outer')
df = df.rename(columns={'What type of image is this?':'pred_cathal',
                        'Quality? (1-5)':'quality_cathal'}).astype(str)
# change label to 0 if contains '/nc/' in string, 1 if contains 'pc_sldct' in string,
# and 2 else
# 0 for full dose nc, 1 for synthetic low dose, 2 for real low dose
df['label_class'] = df['label'].apply(lambda x: 0 if '/nc/' in x else (1 if 'pc_sldct' in x else 2))
df['pred_hania_class'] = df['pred_hania'].apply(lambda x: 1 if 'synthetic' in x else (2 if 'real' in x else 0))
df['pred_cathal_class'] = df['pred_cathal'].apply(lambda x: 1 if 'synthetic' in x else (2 if 'real' in x else 0))
df['pred_iztok_class'] = df['pred_iztok'].apply(lambda x: 0 if 'a' in x else (2 if 'b' in x else 1))

# merge std devs
df = pd.merge(df, stddev, on='StudyID', how='outer')

# merge z_spacing
df = pd.merge(df, im_zspacing, on='StudyID', how='outer')



# confusion matrix
cm_hania = confusion_matrix(df['label_class'], df['pred_hania_class'])
cm_iztok = confusion_matrix(df['label_class'], df['pred_iztok_class'])
cm_cathal = confusion_matrix(df['label_class'], df['pred_cathal_class'])

#create 3x3 confusion matrix that excludes the 80pc_sldct and then the 90pc_sldct
df_80 = df[~df['label'].str.contains('90pc_sldct')]
df_90 = df[~df['label'].str.contains('80pc_sldct')]

cm_80_h = confusion_matrix(df_80['label_class'], df_80['pred_hania_class'])
cm_90_h = confusion_matrix(df_90['label_class'], df_90['pred_hania_class'])

cm_80_i = confusion_matrix(df_80['label_class'], df_80['pred_iztok_class'])
cm_90_i = confusion_matrix(df_90['label_class'], df_90['pred_iztok_class'])

cm_80_c = confusion_matrix(df_80['label_class'], df_80['pred_cathal_class'])
cm_90_c = confusion_matrix(df_90['label_class'], df_90['pred_cathal_class'])

overall_cm_80 = (cm_80_h + cm_80_i + cm_80_c)/(3*88)
overall_cm_90 = (cm_90_h + cm_90_i + cm_90_c)/(3*88)

# fig, ax = plt.subplots(3 , 2   , figsize=(12, 5))
# sns.heatmap(cm_80_h, annot=True, ax=ax[0,0], fmt='d')
# ax[0,0].set_title('Confusion Matrix for 80pc_sldct Hania')
# sns.heatmap(cm_90_h, annot=True, ax=ax[0,1], fmt='d')
# ax[0,1].set_title('Confusion Matrix for 90pc_sldct Hania')
# sns.heatmap(cm_80_i, annot=True, ax=ax[1,0], fmt='d')
# ax[1,0].set_title('Confusion Matrix for 80pc_sldct Iztok')
# sns.heatmap(cm_90_i, annot=True, ax=ax[1,1], fmt='d')
# ax[1,1].set_title('Confusion Matrix for 90pc_sldct Iztok')
# sns.heatmap(cm_80_c, annot=True, ax=ax[2,0], fmt='d')
# ax[2,0].set_title('Confusion Matrix for 80pc_sldct Cathal')
# sns.heatmap(cm_90_c, annot=True, ax=ax[2,1], fmt='d')
# ax[2,1].set_title('Confusion Matrix for 90pc_sldct Cathal')
#
# #add axis titles and labels
# for i in range(3):
#     for j in range(2):
#         ax[i,j].set_xlabel('Predicted')
#         ax[i,j].set_ylabel('True')
#         ax[i,j].set_xticklabels(['Normal Dose', 'Synthetic Low Dose', 'Real Low Dose'])
#         ax[i,j].set_yticklabels(['Normal Dose', 'Synthetic Low Dose', 'Real Low Dose'])

#plot overall confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(overall_cm_80, annot=True, ax=ax[0])
ax[0].set_title('Normalised Confusion Matrix for 80pc_sldct')
sns.heatmap(overall_cm_90, annot=True, ax=ax[1])
ax[1].set_title('Normalised Confusion Matrix for 90pc_sldct')
for i in range(2):
    ax[i].set_xlabel('Predicted')
    ax[i].set_ylabel('True')
    ax[i].set_xticklabels(['Normal Dose', 'Synthetic Low Dose', 'Real Low Dose'])
    ax[i].set_yticklabels(['Normal Dose', 'Synthetic Low Dose', 'Real Low Dose'])

# plot whisker plot of quality scores for each of: full dose, 80pc_sldct, 90pc_sldct, real low dose
df['label_class_q'] = df['label'].apply(lambda x: 'Normal' if '/nc/' in x else ('80' if '80pc_sldct' in x else ('90' if '90pc_sldct' in x else 'real low')))

#print df label_class_q and quality and mean and average quality scores for each of the 4 types of images
df['quality_hania'] = df['quality_hania'].astype(float)
df['quality_iztok'] = df['quality_iztok'].astype(float)
df['quality_cathal'] = df['quality_cathal'].astype(float)
df['average_quality'] = (df['quality_hania'] + df['quality_iztok']+ df['quality_cathal'])/3
print(df[['label_class_q', 'quality_hania']])
print(df.groupby('label_class_q')['quality_hania'].mean())
print(df.groupby('label_class_q')['quality_hania'].std())

plt.savefig(os.path.join(save_loc, 'confusion_matrix.png'))

#plot average quality scores
#check for statistical significance
import scipy.stats as stats
fig, ax = plt.subplots(3, 1, figsize=(12, 12))
from statannot import add_stat_annotation
# sns.boxplot(x='label_class_q', y='average_quality', data=df, ax=ax)

#plot box plots with clusters and significance
sns.boxplot(x='label_class_q', y='average_quality', data=df, ax=ax[0])
#add stats cluster and significance

#change x axis labels
ax[0].set_xticklabels(['Normal Dose', '90% Synthetic Low Dose','80% Synthetic Low Dose', 'Real Low Dose'])

#print average quality scores for each of the 4 types of images
print(df.groupby('label_class_q')['average_quality'].mean())

# add_stat_annotation(ax, data=df, x='label_class_q', y='average_quality', box_pairs=[('Normal', '90'), ('80', '90'), ('80', 'real low'), ('90', 'real low')],
#                     test='t-test_ind', text_format='star', loc='inside', verbose=2,comparisons_correction='bonferroni')
#




# ax.set_xticklabels(['Normal Dose', '80% Synthetic Low Dose','90% Synthetic Low Dose', 'Real Low Dose'])
ax[0].set_xlabel('Type of Image')
ax[0].set_ylabel('Average Quality Score')
ax[0].set_title('Qualitative Quality Scores for Hania, Cathal, and Iztok')
#change x axis labels
ax[0].set_xticklabels(['Normal Dose', '90% Synthetic Low Dose','80% Synthetic Low Dose', 'Real Low Dose'])



#remove x axis label
ax[0].set_xlabel('')
#plot box plots with clusters and significance
sns.boxplot(x='label_class_q', y='stddev', data=df, ax=ax[1])
#add stats cluster and significance

#print average quality scores for each of the 4 types of images
print(df.groupby('label_class_q')['stddev'].mean())


# ax.set_xticklabels(['Normal Dose', '80% Synthetic Low Dose','90% Synthetic Low Dose', 'Real Low Dose'])
ax[1].set_xlabel('Type of Image')
ax[1].set_ylabel('Standard Deviation / HU')
ax[1].set_title('Quantitative Quality Scores for Hania, Cathal, and Iztok')
ax[1].set_xticklabels(['Normal Dose', '90% Synthetic Low Dose','80% Synthetic Low Dose', 'Real Low Dose'])
# set padding between subplots
fig.subplots_adjust(wspace=0.1, hspace=0.4)

# add horizontal grid lines for both subplots
means = []
for type in ['Normal', '80', '90', 'real low']:
    means.append(df[df['label_class_q'] == type]['z_spacing'].mean())

# plot another for z_spacing - a bar chart
sns.barplot(x = ['Normal', '80', '90', 'real low'], y = means, ax=ax[2])

ax[2].set_ylabel('Mean Z-Spacing / mm')
ax[2].set_title('Z Spacing for each type of image')
ax[2].set_xticklabels(['Normal Dose', '90% Synthetic Low Dose','80% Synthetic Low Dose', 'Real Low Dose'])
ax[2].set_xlabel('Type of Image')
# print correlation between z_spacing, quality, and stddev
print(df[['z_spacing', 'average_quality', 'stddev']].corr())
# set y extent
ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[2].set_xlabel('')

plt.savefig(os.path.join(save_loc, 'quality_scores.png'))
plt.show()
#print average z_spacing for each of the 4 types of images
print(df.groupby('label_class_q')['z_spacing'].mean())