import os
os.environ['OV_DATA_BASE'] = "/rds/project/rds-sDuALntK11g"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

spacing = 4
preprocessed_name = '4mm_canceronly_customPP'

lb_classes = [1,2,3]
target_spacing=[spacing]*3
prep = SegmentationPreprocessing(apply_resizing=True,
                                    apply_pooling=False,
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    reduce_lb_to_single_class = True,
                                     window=[-200, 300],
                                     scaling=[100, 0],
                                     pooling_stride=None)

prep.initialise_preprocessing()

prep.preprocess_raw_data(raw_data='final_pairedpretrain_ce',
                          preprocessed_name=preprocessed_name)


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

prep.preprocess_raw_data(raw_data='final_pairedpretrain_nc',
                          preprocessed_name=preprocessed_name)
