import os
os.environ['OV_DATA_BASE'] = "/rds/project/rds-sDuALntK11g"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

spacing = 2
preprocessed_name = '2mm_canceronly'

lb_classes = [1]
target_spacing=[spacing]*3
prep = SegmentationPreprocessing(apply_resizing=True,
                                    apply_pooling=False,
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    reduce_lb_to_single_class = True,
                                     window=[-200, 200],
                                     scaling=[100, 0],
                                     pooling_stride=None)

prep.initialise_preprocessing()

prep.preprocess_raw_data(raw_data='masked_final_pairedpretrain_nc_literallyall',
                          preprocessed_name=preprocessed_name)