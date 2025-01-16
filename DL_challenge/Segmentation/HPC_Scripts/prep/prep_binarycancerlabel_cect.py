import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data/"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

data_name = 'kits23_nooverlap'
spacing = 1

preprocessed_name = '1mm_binary_canceronly'

lb_classes = [2]
target_spacing=[spacing]*3
kits33_nooverlap_canc_properties = {'median_shape': [103., 512., 512.],
                                'median_spacing': [4., 0.78125, 0.78125],
                                'fg_percentiles' : [-500.,  281.],
                                'percentiles': [0.5, 99.5],
                                'scaling_foreground': [ 234.7703,  -375.88345],
                                'n_fg_classes': 1,
                                'scaling_global': [ 123.38048, -470.74335],
                                'scaling_window' : [ 121.11153, -470.54962]}

prep = SegmentationPreprocessing(apply_resizing=True, 
                                    apply_pooling=False, 
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                 window = [-500, 281],
                                 scaling=[234.7703, -375.88345],
                                 pooling_stride=None,
                                    target_spacing=target_spacing,
                                    reduce_lb_to_single_class = True)

prep.plan_preprocessing_raw_data(raw_data=data_name)

prep.preprocess_raw_data(raw_data=data_name,
                          preprocessed_name=preprocessed_name)