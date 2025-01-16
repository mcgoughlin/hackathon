import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import sys
data_name = 'test_set_v2'
spacings = [1,2]

for spacing in spacings:
    preprocessed_name = '{}mm_testprep_forradiologist'.format(spacing)
    target_spacing=[spacing]*3
    prep = SegmentationPreprocessing(apply_resizing=True,
                                        apply_pooling=False,
                                        apply_windowing=True,
                                        lb_classes=None,
                                        target_spacing=target_spacing,
                                        scaling = [1, 0],
                                        window = [-1024,1024],
                                        reduce_lb_to_single_class = True,
                                        pooling_stride=None)

    prep.initialise_preprocessing()
    prep.preprocess_raw_data(raw_data=data_name,
                              preprocessed_name=preprocessed_name)