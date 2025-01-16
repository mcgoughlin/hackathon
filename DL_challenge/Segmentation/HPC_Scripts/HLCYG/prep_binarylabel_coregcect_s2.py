import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
#import dilation stuff
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import sys

dose = float(sys.argv[1])
masked_data_name = 'masked_coreg_v3_HLCYG_{}'.format(dose)

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
prep.preprocess_raw_data(raw_data=masked_data_name,
                          preprocessed_name=preprocessed_name)