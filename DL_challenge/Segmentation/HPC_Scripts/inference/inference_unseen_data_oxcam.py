import os
os.environ['OV_DATA_BASE'] = '/rds/project/rds-sDuALntK11g/'

from KCD.Segmentation.Inference.SingleModelSeg import Ensemble_Seg
import numpy as np

seg_fp = '/rds/project/rds-sDuALntK11g/trained_models/kits_nooverlap_v2/2mm_kidneybinary_customPP/6_tot/'
# seg_fp = '/home/wcm23/rds/hpc-work/FineTuningKITS23/trained_models/all_cect/4mm_binary/6,3x3x3,32'
data_names = ['oxcam']
do_prep = False
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(data_name, name='kidney',
                        seg_fp=seg_fp,
                        spacing=np.array([2] * 3),
                        do_prep=do_prep, do_infer=do_infer,
                        is_cect=True, cont=False,overlap=0.5,
                        batch_size=4)

seg_fp = '/rds/project/rds-sDuALntK11g/trained_models/kits23_add_nooverlap/6,3x3x3,32_justcancer_trainontest'
# seg_fp = '/home/wcm23/rds/hpc-work/FineTuningKITS23/trained_models/all_cect/4mm_binary/6,3x3x3,32'
data_names = ['oxcam']
do_prep = True
do_infer = True

for data_name in data_names:
    test = Ensemble_Seg(data_name, name='cancer',
                        seg_fp=seg_fp,
                        spacing=np.array([2] * 3),
                        do_prep=do_prep, do_infer=do_infer,
                        is_cect=True, cont=False,overlap=0.5,
                        batch_size=4)

