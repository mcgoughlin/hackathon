import os
import pandas as pd
import pickle

results_home = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/masked_coreg_v2_noised/2mm_cancerbinary_customPP'

#extract all folders in results_home

folders = [f for f in os.listdir(results_home) if os.path.isdir(os.path.join(results_home, f))]

# for each folder, extract the validation results
results = []
for folder in folders:
    print(folder)
    #check if 'validation_CV_results.pkl' exists
    results_path = os.path.join(results_home, folder, 'validation_CV_results.pkl')

    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
            df = pd.DataFrame(data).T
            #check columns excist in df, if not skip
            if 'average_detection_0.01' not in df.columns:
                continue
            average_detection_001 = df['average_detection_0.01'].mean()
            average_detection_010 = df['average_detection_0.1'].mean()
            average_detection_050 = df['average_detection_0.5'].mean()
            average_detection_091 = df['average_detection_0.91'].mean()
            average_detection_099 = df['average_detection_0.99'].mean()


            entry = {'model':folder,
                     'average_detection_001':average_detection_001,
                     'average_detection_010':average_detection_010,
                     'average_detection_050':average_detection_050,
                     'average_detection_091':average_detection_091,
                     'average_detection_099':average_detection_099}

            print(entry)
            results.append(entry)

results_df = pd.DataFrame(results)
# convert all average columns to floats with 3 decimals
results_df['average_detection_001'] = results_df['average_detection_001'].apply(lambda x: round(x, 3)).astype(float)
results_df['average_detection_010'] = results_df['average_detection_010'].apply(lambda x: round(x, 3)).astype(float)
results_df['average_detection_050'] = results_df['average_detection_050'].apply(lambda x: round(x, 3)).astype(float)
results_df['average_detection_091'] = results_df['average_detection_091'].apply(lambda x: round(x, 3)).astype(float)
results_df['average_detection_099'] = results_df['average_detection_099'].apply(lambda x: round(x, 3)).astype(float)

results_df.to_csv(os.path.join(results_home, 'overall_results.csv'))
