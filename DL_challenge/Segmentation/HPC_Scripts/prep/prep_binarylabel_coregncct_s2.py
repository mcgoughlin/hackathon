import os
os.environ['OV_DATA_BASE'] = "/home/wcm23/rds/hpc-work/FineTuningKITS23"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import sys
data_name = 'masked_coreg_v4_noised'
spacing = 2

preprocessed_name = '{}mm_cancerbinary_customPP'.format(spacing)
lb_classes = [2]
target_spacing=[spacing]*3
prep = SegmentationPreprocessing(apply_resizing=True, 
                                    apply_pooling=False, 
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    scaling = [100, 0],
                                    window = [-200,200],
                                    reduce_lb_to_single_class = True,
                                    pooling_stride=None)
prep.initialise_preprocessing()
prep.preprocess_raw_data(raw_data=data_name,
                          preprocessed_name=preprocessed_name)

#copy 'splits.pkl' from dir_a to dir_b

dir_a = '/home/wcm23/rds/hpc-work/FineTuningKITS23/preprocessed/coreg_v4_noised/4mm_kidneybinary_customPP'
dir_b = '/home/wcm23/rds/hpc-work/FineTuningKITS23/preprocessed/masked_coreg_v4_noised/2mm_cancerbinary_customPP'

import shutil
shutil.copy(os.path.join(dir_a, 'splits.pkl'), dir_b)