import os
os.environ['OV_DATA_BASE'] = '/rds/project/rds-sDuALntK11g/'

from KCD.Segmentation.Inference.SingleModelSeg import Ensemble_Seg
import numpy as np

seg_fp = '/rds/project/rds-sDuALntK11g/2mm_kidneybinary_customPP/6_tot'
# seg_fp = '/home/wcm23/rds/hpc-work/FineTuningKITS23/trained_models/all_cect/4mm_binary/6,3x3x3,32'
data_names = ['final_pairedpretrain_ce']
do_prep = True
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(data_name,
                        seg_fp=seg_fp,
                        spacing=np.array([2] * 3),
                        do_prep=do_prep, do_infer=do_infer,
                        is_cect=True, cont=False,overlap=0.5,
                        batch_size=4, patch_size=64)

