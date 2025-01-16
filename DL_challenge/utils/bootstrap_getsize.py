import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.measure import label
import pandas as pd
import sys

model = '6_finetune_l2-1e-05_cce1.0_cos1.0_2e-05lr_0.999997lg_16bs_2000ep_4gpus_efr_dfr100_seg_lr0.0001'

home = '/bask/projects/p/phwq4930-renal-canc/data/seg_data'
labels = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/masked_coreg_v2_noised/labels/'
preds = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions/masked_coreg_v2_noised/2mm_cancerbinary_customPP/{}/cross_validation_continuous/'.format(model)
new_fold = os.path.join(home, 'conv_intv', model)
csv_path = os.path.join(new_fold, 'sizes.csv')

if not os.path.exists(new_fold):
    os.makedirs(new_fold)

#save csv of file, size of max connected component

files = [f for f in os.listdir(preds) if f.endswith('.nii.gz')]
results = []
for file in files:
    lab = nib.load(os.path.join(labels, file.replace('_0.', '_0.7.')))
    spacing = lab.header.get_zooms()
    conn_comp = label(lab.get_fdata()==2)
    sizes = [np.sum(conn_comp==i) for i in range(1, np.max(conn_comp)+1)]
    size = max(sizes) * np.prod(spacing)
    print(file,size)
    results.append({'file':file, 'size':size})

df = pd.DataFrame(results)
df.to_csv(csv_path)

