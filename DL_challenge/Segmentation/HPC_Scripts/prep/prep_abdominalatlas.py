import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import sys

data_name = 'abdominal_atlas'
spacing = int(sys.argv[1])

preprocessed_name = '{}mm'.format(spacing)

lb_classes = [1,2,3,4,5,6,7,8,9]
target_spacing=[spacing]*3

# dataset_properties =
# median_shape = [169. 492. 370.]
# median_spacing = [2.5       0.8144531 0.8144531]
# fg_percentiles = [-963.1221374   279.09160305]
# percentiles = [0.5, 99.5]
# scaling_foreground = [inf 68.0625]
# n_fg_classes = 9
# scaling_global = [477.88907 - 376.74026]
# scaling_window = [460.5284 - 373.891]

dataset_properties = {
    'median_shape': [169, 492, 370],
    'median_spacing': [2.5, 0.8144531, 0.8144531],
    'fg_percentiles': [-279.09, 279.09160305],
    'percentiles': [0.5, 99.5],
    'scaling_foreground': [55.6, 68.0625],
    'n_fg_classes': 9,
    'scaling_global': [477.88907, -376.74026],
    'scaling_window': [460.5284, -373.891]
}

prep = SegmentationPreprocessing(apply_resizing=True,
                                    apply_pooling=False, 
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    reduce_lb_to_single_class = False,
                                    dataset_properties=dataset_properties,
                                 pooling_stride=None,
                                 window = [-279.09, 279.09160305],
                                 scaling=[55.6, 68.0625],
                                 )

prep.initialise_preprocessing()

prep.preprocess_raw_data(raw_data=data_name,
                          preprocessed_name=preprocessed_name)