import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from skimage.measure import label
import pandas as pd
import sys


dose = float(sys.argv[1])

model = 'HLCYG_{}_2'.format(dose)
data_name = 'AImasked_coreg_v2_HLCYGnoiseonly_{}'.format(dose)

home = '/bask/projects/p/phwq4930-renal-canc/data/seg_data'
labels = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/AImasked_coreg_ncct_v2/labels/'
preds = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions/{}/2mm_cancerbinary_customPP/{}/cross_validation_continuous/'.format(data_name,model)
new_fold = os.path.join(home, 'conv_intv', data_name)
save_loc = os.path.join(new_fold, 'conf_ROC.npy')
ppc_path = os.path.join(new_fold, 'conf_per_patient.npy')
csv_path = os.path.join(new_fold, 'conf_ROC.csv')
png_path = os.path.join(new_fold, 'conf_ROC.png')
sizes_path = os.path.join(new_fold, 'sizes.csv')


if not os.path.exists(new_fold):
    os.makedirs(new_fold)


confidences = [0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
               0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.995,0.999,0.9999]



files = [f for f in os.listdir(preds) if f.endswith('.nii.gz')]
per_kidney_confidences = {}

results = [[1, 0]]
sizes = []
per_patient_confidences = np.zeros((len(files), len(confidences),4))
for x,conf in enumerate(confidences):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for y,file in enumerate(files):
        patient_tptnfpfn = np.zeros((4))

        label_nib = nib.load(os.path.join(labels, file.replace('_noised.','.')))
        pred_nib = nib.load(os.path.join(preds, file))
        seg = label_nib.get_fdata()
        pred = pred_nib.get_fdata()
        voxvol = np.prod(label_nib.header.get_zooms())

        label_foreground = (seg ==2).astype(int)
        if len(sizes) != len(files):
            # find max size of connected component
            max_size = 0
            for i in range(1, np.max(label_foreground) + 1):
                component = label_foreground == i
                if np.sum(component) > max_size:
                    max_size = np.sum(component)
            sizes.append((file, max_size))

        pred_foreground = (pred > conf).astype(int)
        pred_components = label(np.squeeze(pred_foreground))
        label_components = label(np.squeeze(label_foreground))
        # find the centroids of each prediction
        pred_pos_centroids = []
        # save the size of the kidney
        for i in range(1, np.max(pred_components) + 1):
            pred_component = pred_components == i
            if pred_component.sum() < (500 / voxvol):
                continue
            pred_pos_centroid = np.round(np.mean(np.where(pred_component), axis=1)).astype(int)
            pred_pos_centroids.append(pred_pos_centroid)
        pred_pos_centroids = np.array(pred_pos_centroids)

        per_kidney_confidences[file] = [[0, 0]] * len(pred_pos_centroids)

        ### THIS SECTION ASSESSES PREDICTIONS
        kidney_binary = (seg > 0).astype(int)
        kidney_components = label(np.squeeze(kidney_binary))
        kidney_cancer = np.zeros_like(kidney_binary)
        # get rid of any connected components in kidney_binary that contains a positive prediction
        for i in range(1, np.max(label_components) + 1):
            label_component = label_components == i
            label_pos_centroid = np.round(np.mean(np.where(label_component), axis=1)).astype(int)
            kidney_component = kidney_components == kidney_components[label_pos_centroid[0], label_pos_centroid[1], label_pos_centroid[2]]
            #ignore small components
            kidney_cancer[kidney_component] = 1


        kidney_healthy = kidney_binary - kidney_cancer
        kidney_cancer_checked = np.zeros_like(kidney_cancer)

        # count number of positive predictions in kidney_negative and kidney_positive - these are false and true positives

        for pred_pos_centroid in pred_pos_centroids:
            if kidney_cancer_checked[pred_pos_centroid[0], pred_pos_centroid[1], pred_pos_centroid[2]] == 1:
                continue
            elif kidney_cancer[pred_pos_centroid[0], pred_pos_centroid[1], pred_pos_centroid[2]] == 1:
                tp += 1
                #get rid of the kidney_cancer component that contains this positive prediction
                kidney_component = kidney_components == kidney_components[pred_pos_centroid[0], pred_pos_centroid[1], pred_pos_centroid[2]]
                # print(np.round(np.mean(np.where(kidney_component), axis=1)).astype(int), 'tp')
                # print('Number of voxels:', np.sum(kidney_component))
                kidney_cancer_checked[kidney_component] = 1
                patient_tptnfpfn[0] += 1
            else:
                # this counts non-kidney predictions as false positives, too
                fp += 1
                kidney_component = kidney_components == kidney_components[pred_pos_centroid[0], pred_pos_centroid[1], pred_pos_centroid[2]]
                # print(np.round(np.mean(np.where(kidney_component), axis=1)).astype(int), 'fp')
                # print('Number of voxels:', np.sum(kidney_component))
                kidney_cancer_checked[kidney_component] = 1
                patient_tptnfpfn[2] += 1

        ### THIS SECTIONS ASSESSES OMITTED PREDICTIONS

        kidney_positive = np.zeros_like(kidney_binary)
        kidney_binary = (seg > 0).astype(int) - kidney_cancer_checked
        new_kidney_components = label(np.squeeze(kidney_binary))
        # print(np.max(new_kidney_components))
        # check if any positive predictions overlap each kidney region
        for pred_pos_centroid in pred_pos_centroids:
            kidney_component = new_kidney_components == new_kidney_components[pred_pos_centroid[0], pred_pos_centroid[1], pred_pos_centroid[2]]
            kidney_positive[kidney_component] = 1

        kidney_negative = kidney_binary - kidney_positive

        # check each kidney_negative component - if it exists in kidney_healthy, it is a true negative,
        # otherwise it is a false negative
        kidney_components = label(np.squeeze(kidney_negative))
        for i in range(1, np.max(kidney_components) + 1):
            kidney_component = kidney_components == i
            if kidney_cancer_checked[kidney_component].sum() > 0:
                continue
            if np.sum(kidney_healthy[kidney_component]) > 0:
                #print centroid of kidney_component
                # print(np.round(np.mean(np.where(kidney_component), axis=1)).astype(int), 'tn')
                tn += 1
                kidney_cancer_checked[kidney_component] = 1
                patient_tptnfpfn[1] += 1
            elif np.sum(kidney_cancer[kidney_component]) > 0:
                # print(np.round(np.mean(np.where(kidney_component), axis=1)).astype(int), 'fn')
                fn += 1
                kidney_cancer_checked[kidney_component] = 1
                patient_tptnfpfn[3] += 1
            else:
                print('Error')
                print(np.sum(kidney_component))
                pass

        per_patient_confidences[y,x] = patient_tptnfpfn

    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9)
    npv = tn / (tn + fn + 1e-9)
    print(f'Confidence: {conf}, Sens: {sens}, Spec: {spec}, PPV: {ppv}, NPV: {npv}')
    print()
    sys.stdout.flush()
    results.append([sens, spec])
    results = np.array(results)
    print('1')
    print()
    sys.stdout.flush()
    np.save(save_loc, results)
    print('2')
    print()
    sys.stdout.flush()

    np.save(ppc_path, per_patient_confidences)
    print('3')
    print()
    sys.stdout.flush()

    df = pd.DataFrame(np.concatenate([results,[[0,1]]],axis=0), columns=['Sensitivity', 'Specificity'])
    df.to_csv(csv_path)

    df['1-Specificity'] = 1 - df['Specificity']

    AUC = np.trapz(df['Sensitivity'], df['1-Specificity'])
    plt.plot(df['1-Specificity'], df['Sensitivity'])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'AUC: {AUC}')
    plt.savefig(png_path)
    plt.show()
    results = results.tolist()


results.append([0,1])
results = np.array(results)
print(results,'a')
print()
sys.stdout.flush()
np.save(save_loc, results)
print(per_patient_confidences,'b')
print()
sys.stdout.flush()
np.save(ppc_path, per_patient_confidences)
size_df = pd.DataFrame(sizes, columns=['File', 'Size'])
size_df.to_csv(sizes_path)

df = pd.DataFrame(results, columns=['Sensitivity', 'Specificity'])
df['Confidence'] = np.concatenate([[0], confidences, [1]])
df.to_csv(csv_path)


df['1-Specificity'] = 1 - df['Specificity']

AUC = np.trapz(df['Sensitivity'], df['1-Specificity'])
plt.plot(df['1-Specificity'], df['Sensitivity'])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title(f'AUC: {AUC}')
plt.savefig(png_path)
plt.show()
