import os
os.environ['OV_DATA_BASE'] = '/bask/projects/p/phwq4930-renal-canc/data/seg_data'

from KCD.Segmentation.Inference.SingleModelSeg_hania import Ensemble_Seg
import numpy as np

seg_fp = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/abdominal_atlas/liver_2/7_tot_liver_longer'
data_names = ['btcv']
do_prep = True
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(data_name, name='hania',
                        seg_fp=seg_fp, patch_size=128,
                        spacing=np.array([2] * 3),
                        do_prep=do_prep, do_infer=do_infer,
                        cont=False,overlap=0.5,
                        batch_size=4)