import os
import pandas as pd
import numpy as np

results_home = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/conv_intv'

#extract all folders in results_home

folders = [f for f in os.listdir(results_home) if os.path.isdir(os.path.join(results_home, f))]

# for each folder, extract the validation results
weight = 4
results = []
for folder in folders:
    print(folder)
    #check if 'validation_CV_results.pkl' exists
    results_path = os.path.join(results_home, folder, 'conf_ROC.csv')

    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            df = pd.read_csv(f)
            print(df.head())
            sens = df['Sensitivity'].values
            spec = df['Specificity'].values
            conf = df['Confidence'].values
            high_acc,op_acc = 0,0
            high_sens,op_sens = 0,0
            high_spec,op_spec = 0,0

            for i in range(len(conf)):
                acc_score = (sens[i] + spec[i])/2
                sens_score = (weight*sens[i] + spec[i])/(weight+1)
                spec_score = (weight*spec[i] + sens[i])/(weight+1)
                if acc_score > high_acc:
                    high_acc = acc_score
                    op_acc = conf[i]
                if sens_score > high_sens:
                    high_sens = sens_score
                    op_sens = conf[i]
                if spec_score > high_spec:
                    high_spec = spec_score
                    op_spec = conf[i]

            AUC = np.trapz(sens, spec)

            results.append({'model':folder,
                            'high_acc':high_acc,
                            'op_acc':op_acc,
                            'high_sens':high_sens,
                            'op_sens':op_sens,
                            'high_spec':high_spec,
                            'op_spec':op_spec,
                            'AUC':AUC})

df = pd.DataFrame(results)
#convert all columns to floats with 3 decimals
df['high_acc'] = df['high_acc'].apply(lambda x: round(x, 4)).astype(float)
df['op_acc'] = df['op_acc'].apply(lambda x: round(x, 4)).astype(float)
df['high_sens'] = df['high_sens'].apply(lambda x: round(x, 4)).astype(float)
df['op_sens'] = df['op_sens'].apply(lambda x: round(x, 4)).astype(float)
df['high_spec'] = df['high_spec'].apply(lambda x: round(x, 4)).astype(float)
df['op_spec'] = df['op_spec'].apply(lambda x: round(x, 4)).astype(float)
df['AUC'] = df['AUC'].apply(lambda x: round(x, 4)).astype(float)

print(df)

df.to_csv(os.path.join(results_home, 'overall_results.csv'))

# time to create paper-ready table

from_trained_filt = df[df['model'].str.contains('fromtrained_filt')]
from_trained = df[df['model'].str.contains('fromtrained') & ~df['model'].str.contains('filt')]
from_onehead = df[df['model'].str.contains('cos0.0')]
from_twohead = df[df['model'].str.contains('cos1.0') & df['model'].str.contains('l2-0.0')]
from_threehead = df[df['model'].str.contains('cos1.0') & df['model'].str.contains('l2-1')]

#print length of each one
print(len(from_trained), len(from_onehead), len(from_twohead), len(from_threehead))

final_results = []
#iterate through rows of from_trained
for index, row in from_trained.iterrows():
    AUC = row['AUC']
    model = row['model']
    #split model string by underscore
    model = model.split('_')
    # if the final element is a decimal, it is the learning rate
    if model[-1].isdigit():
        lr = 0.00005
        dfr = model[-1]
        efr = model[-2]
    else:
        lr = model[-1]
        dfr = model[-2]
        efr = model[-3]

    final_results.append({'AUC':float(AUC),
                                'lr':float(lr),
                                'dfr':int(dfr),
                                'efr':int(efr),
                          'model_origin':'from_trained'})


for index, row in from_trained_filt.iterrows():
    AUC = row['AUC']
    model = row['model']
    #split model string by underscore
    model = model.split('_')
    # if the final element is a decimal, it is the learning rate
    if model[-1].isdigit():
        lr = 0.00005
        dfr = model[-1]
        efr = model[-2]
    else:
        lr = model[-1]
        dfr = model[-2]
        efr = model[-3]

    final_results.append({'AUC':float(AUC),
                                'lr':float(lr),
                                'dfr':int(dfr),
                                'efr':int(efr),
                          'model_origin':'from_trained_filt'})



#iterate through rows of from_onehead
for index, row in from_onehead.iterrows():
    AUC = row['AUC']
    model = row['model']
    #split model string by underscore
    model = model.split('_')
    # if the final element is a number, it is the learning rate

    if model[-1] == 'seg':
        lr = 0.00005
        dfr = model[-2][3:]
        efr = model[-3][3:]
    else:
        lr = model[-1][2:]
        dfr = model[-3][3:]
        efr = model[-4][3:]

    if efr == '':
        efr = 1000
    if dfr == '':
        dfr = 1000
    print('Learning rate:', lr, 'dfr:', dfr, 'efr:', efr)

    final_results.append({'AUC':float(AUC),
                                'lr':float(lr),
                                'dfr':int(dfr),
                                'efr':int(efr),
                          'model_origin':'from_onehead'})

#iterate through rows of from_twohead - same as from_onehead
for index, row in from_twohead.iterrows():
    AUC = row['AUC']
    model = row['model']
    #split model string by underscore
    model = model.split('_')
    # if the final element is a number, it is the learning rate

    if model[-1] == 'seg':
        lr = 0.00005
        dfr = model[-2][3:]
        efr = model[-3][3:]
    else:
        lr = model[-1][2:]
        dfr = model[-3][3:]
        efr = model[-4][3:]

    if efr == '':
        efr = 1000
    if dfr == '':
        dfr = 1000
    print('Learning rate:', lr, 'dfr:', dfr, 'efr:', efr)

    final_results.append({'AUC':float(AUC),
                                'lr':float(lr),
                                'dfr':int(dfr),
                                'efr':int(efr),
                          'model_origin':'from_twohead'})

#iterate through rows of from_threehead - same as from_onehead
for index, row in from_threehead.iterrows():
    AUC = row['AUC']
    model = row['model']
    #split model string by underscore
    model = model.split('_')
    # if the final element is a number, it is the learning rate

    if model[-1] == 'seg':
        lr = 0.00005
        dfr = model[-2][3:]
        efr = model[-3][3:]
    else:
        lr = model[-1][2:]
        dfr = model[-3][3:]
        efr = model[-4][3:]

    if efr == '':
        efr = 1000
    if dfr == '':
        dfr = 1000
    print('Learning rate:', lr, 'dfr:', dfr, 'efr:', efr)

    final_results.append({'AUC':float(AUC),
                                'lr':float(lr),
                                'dfr':int(dfr),
                                'efr':int(efr),
                          'model_origin':'from_threehead'})

for entry in final_results:
    print(entry)

final_df = pd.DataFrame(final_results)
final_df.sort_values(['model_origin','lr','dfr','efr'], inplace=True)
final_df['lr'] = final_df['lr'].astype(float)
#get rid of rows where lr is less than 0.00005
final_df = final_df[final_df['lr'] >= 0.00005]


print(final_df)
final_df.to_csv(os.path.join(results_home, 'final_results.csv'))
