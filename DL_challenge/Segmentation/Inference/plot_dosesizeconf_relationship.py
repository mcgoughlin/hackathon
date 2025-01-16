#combine casewise_confidences.csv with results.csv - add a column for dose in casewise_confidences.csv

import pandas
import os

casewise_confidences = pandas.read_csv('/Users/mcgoug01/Downloads/test_results/casewise_confidences.csv')
results = pandas.read_csv('/Users/mcgoug01/Downloads/test_results/output/results.csv')

casewise_confidences['patient'] = casewise_confidences['file'].str.split('.').str[0]

casewise_confidences = casewise_confidences.merge(results, on='patient', how='left')
print(casewise_confidences.columns)
#remove any column with 'Unnamed' and 'patient
casewise_confidences = casewise_confidences.loc[:, ~casewise_confidences.columns.str.contains('^Unnamed')]
# drop where label is 0
casewise_confidences['radius_mm'] = (casewise_confidences['size_mm']*(3/4)/3.14159)**(1/3)
casewise_confidences['dose'] = casewise_confidences['dose']*(8.6/225)
casewise_confidences['pred'] = casewise_confidences['confidence'] > 0.5
casewise_confidences['correct'] = casewise_confidences['pred'] == casewise_confidences['label']
#find sensitivity above and below 20mm radius, and above/below dose of 100
sensitivities = []
small_sensitivities = []
for boundary_radius in [0,3,5, 10, 15, 20, 25, 30, 35, 40,60]:
    casewise_confidences[f'below_{boundary_radius}mm'] = casewise_confidences['radius_mm'] < boundary_radius
    sensitivity_radius = casewise_confidences[casewise_confidences[f'below_{boundary_radius}mm'] & (casewise_confidences['label']==1)]['correct'].mean() * 100
    sensitivities.append({'boundary_radius': boundary_radius,'sensitivity_radius': sensitivity_radius})



for boundary_dose in [50,60, 75, 100, 125, 150, 175, 200,300]:
    boundary_dose = boundary_dose * 8.6/225
    casewise_confidences[f'below_{boundary_dose}mGy'] = casewise_confidences['dose'] < boundary_dose
    sensitivity_dose = casewise_confidences[casewise_confidences[f'below_{boundary_dose}mGy'] & (casewise_confidences['label']==1)]['correct'].mean() * 100
    small_sensitivity_dose = casewise_confidences[casewise_confidences[f'below_{boundary_dose}mGy'] & (casewise_confidences['label']==1) & (casewise_confidences['radius_mm']<20)]['correct'].mean() * 100
    sensitivities.append({'boundary_dose': boundary_dose, 'sensitivity_dose': sensitivity_dose})

print(casewise_confidences.head(), casewise_confidences.columns)
for boundary_dose in [ 50,100,150,200,250,300,500]:
    boundary_dose = boundary_dose * 8.6 / 225
    casewise_confidences[f'below_{boundary_dose}mGy'] = casewise_confidences['dose'] < boundary_dose
    print(f'Number of cancers below {boundary_dose}mGy: {casewise_confidences[casewise_confidences[f"below_{boundary_dose}mGy"]]["label"].sum()}')
    #print the cases file names of small cancers that labelled as cancer and below dose
    num_small_cases = casewise_confidences[casewise_confidences[f"below_{boundary_dose}mGy"] & (casewise_confidences["radius_mm"]<20)]["label"].sum()
    print(f'Number of small cancers below {boundary_dose}mGy: {num_small_cases}')
    # print the pred and correct columns of these cases
    print(casewise_confidences[casewise_confidences[f'below_{boundary_dose}mGy'] & (casewise_confidences['label']==1) & (casewise_confidences['radius_mm']<20)][['pred', 'correct']])
    small_sensitivity_dose = casewise_confidences[casewise_confidences[f'below_{boundary_dose}mGy'] & (casewise_confidences['label']==1) & (casewise_confidences['radius_mm']<20)]['correct'].mean() * 100
    print(f'Sensitivity for small cancers below {boundary_dose}mGy: {small_sensitivity_dose}\n')
    small_sensitivities.append({'boundary_dose': boundary_dose, 'small_sensitivity_dose': small_sensitivity_dose, 'num_small_cases': num_small_cases})

sensitivities = pandas.DataFrame(sensitivities)
small_sensitivities = pandas.DataFrame(small_sensitivities)
sensitivities.to_csv('/Users/mcgoug01/Downloads/test_results/sensitivities.csv')
small_sensitivities.to_csv('/Users/mcgoug01/Downloads/test_results/small_sensitivities.csv')

specificity = casewise_confidences[casewise_confidences['label']==0]['correct'].mean()
print(specificity)


import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('TkAgg')

correlation_radconf = casewise_confidences[casewise_confidences['label']==1][['radius_mm', 'confidence']].corr()
correlation_doseconf = casewise_confidences[casewise_confidences['label']==1][['dose', 'confidence']].corr()

# plot a two 2d scatter plot of size and and dose vs confidence. confidence on y axis, size and dose on x axis
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=casewise_confidences[casewise_confidences['label']==1], y='confidence', x='radius_mm', ax=ax[0])
sns.scatterplot(data=casewise_confidences[casewise_confidences['label']==1], y='confidence', x='dose', ax=ax[1])

#draw a boundary at 20mm radius and 100mAs
ax[0].axvline(20, color='r')
ax[1].axvline(3.82, color='r')

#put text left of boundary saying 'small' and 'low dose'
ax[0].text(5, 0.5, 'Small', color='r', fontsize=12)
ax[0].text(40, 0.5, 'Large', color='g', fontsize=12)
ax[1].text(1.5, 0.5, 'Low\nDose', color='r', fontsize=12)
ax[1].text(8, 0.5, 'High\nDose', color='g', fontsize=12)

#put text in top right of each plot with correlation - like a legend
ax[0].text(1.5, 0.9, f'Correlation: {correlation_radconf.iloc[0,1]:.2f}', fontsize=12)
ax[1].text(8, 0.9, f'Correlation: {correlation_doseconf.iloc[0,1]:.2f}', fontsize=12)
plt.savefig('/Users/mcgoug01/Downloads/Figure_1.png')
plt.show()


plt.close()

#plot two curves of sensitivity vs dose and sensitivity vs size
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.lineplot(data=sensitivities, x='boundary_dose', y='sensitivity_dose', ax=ax[0],linestyle=':')
# sns.lineplot(data=sensitivities, x='boundary_radius', y='sensitivity_radius', ax=ax[1]), with a legend name of 'Sensitivity'
sns.lineplot(data=small_sensitivities, x='boundary_dose', y='small_sensitivity_dose', ax=ax[1],linestyle=':')
#plot a secondary axis on the right of the plot with the number of small cancers, with a linestyle of dotted
ax2 = ax[1].twinx()
sns.scatterplot(data=small_sensitivities, x='boundary_dose', y='num_small_cases', ax=ax2, color='k',label='Number of Small Cancers')
ax2.set_ylabel('Number of Small Cancers')

#change ax[1] exten from 50 to 100 on y
ax[1].set_ylim(0, 100)
ax[0].set_ylim(0, 100)



# add a dotted red line at dose of 4.0 with a label 'YKST median dose'
# and a dotted black line at 2.0 for dose with label 'YLST median dose'
ax[0].axvline(4.0, color='r', linestyle='--', label='YKST median dose')
ax[0].axvline(2.0, color='k', linestyle='--', label='YLST median dose')
ax[1].axvline(4.0, color='r', linestyle='--')
ax[1].axvline(2.0, color='k', linestyle='--')

ax[0].legend()

ax[0].set_title('Sensitivity vs Dose (all cancers)')
# ax[1].set_title('Sensitivity vs Size')
ax[1].set_title('Sensitivity vs Dose (Small Cancers)')

ax[0].set_xlabel('Dose (mGy)')
# ax[1].set_xlabel('Radius of Cancer (mm)')
ax[0].set_ylabel('Sensitivity / %')
# ax[1].set_ylabel('Sensitivity / %')
ax[1].set_xlabel('Dose (mGy)')
ax[1].set_ylabel('Sensitivity')
# make a manual legend on ax[1] for ykst, ylst, sensitivity, and number of small cancers
ax[1].legend(['Sensitivity', 'Number of Small Cancers', 'YKST median dose', 'YLST median dose'])

correlation = casewise_confidences[['radius_mm', 'dose']].corr()
# ax[0].text(3.8, 60, f'Correlation between\nradius and dose: {correlation.iloc[0,1]:.2f}', fontsize=12)
plt.savefig('/Users/mcgoug01/Downloads/Figure_2.png')
plt.show()


plt.close()

# plot a histogram of doses
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.histplot(casewise_confidences['dose'], ax=ax, bins=20)
ax.set_title('Histogram of Doses')
ax.set_xlabel('Dose (mGy)')
ax.set_ylabel('Frequency')
plt.savefig('/Users/mcgoug01/Downloads/Figure_3.png')
plt.show()


casewise_confidences.to_csv('/Users/mcgoug01/Downloads/test_results/dose_and_size.csv')