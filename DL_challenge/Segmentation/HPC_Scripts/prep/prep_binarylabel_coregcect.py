import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import sys
data_name = 'coreg_cect_v3'
spacing = 4

preprocessed_name = '{}mm_kidneybinary_customPP'.format(spacing)
lb_classes = [1,2,3]
target_spacing=[spacing]*3
prep = SegmentationPreprocessing(apply_resizing=True, 
                                    apply_pooling=False, 
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    scaling = [100, 0],
                                    window = [-200,300],
                                    reduce_lb_to_single_class = True,
                                    pooling_stride=None)
prep.initialise_preprocessing()
prep.preprocess_raw_data(raw_data=data_name,
                          preprocessed_name=preprocessed_name)