import os
os.environ['OV_DATA_BASE'] = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/'

from KCD.Segmentation.Inference.EnsembleSeg import Ensemble_Seg
import numpy as np

seg_fp = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/coreg_v2_noised/4mm_kidneybinary_customPP/6_finetune'
# seg_fp = '/home/wcm23/rds/hpc-work/FineTuningKITS23/trained_models/all_cect/4mm_binary/6,3x3x3,32'
data_names = ['coreg_v2_noised']
do_prep = True
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(data_name,name='cect',
                        seg_fp=seg_fp,
                        spacing=np.array([4] * 3),
                        do_prep=do_prep, do_infer=do_infer,
                        is_cect=False, cont=False,overlap=0.5,
                        batch_size=4, patch_size=64)


seg_fp = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/coreg_v2_noised/4mm_kidneybinary_customPP/6_finetune_l2-0.0_cce1.0_2e-05lr_0.999993lg_16bs_1000ep_4gpus'
# seg_fp = '/home/wcm23/rds/hpc-work/FineTuningKITS23/trained_models/all_cect/4mm_binary/6,3x3x3,32'
data_names = ['coreg_v2_noised']
do_prep = True
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(data_name,name = 'radar',
                        seg_fp=seg_fp,
                        spacing=np.array([4] * 3),
                        do_prep=do_prep, do_infer=do_infer,
                        is_cect=False, cont=False,overlap=0.5,
                        batch_size=4, patch_size=64)

