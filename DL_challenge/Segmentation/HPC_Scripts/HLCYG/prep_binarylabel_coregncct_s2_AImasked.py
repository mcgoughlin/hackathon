import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
#import dilation stuff
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import sys

dose = float(sys.argv[1])

data_name = 'coreg_v2_HLCYG_'+str(dose)
masked_data_name = 'AImasked_'+data_name
raw_data_location = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', data_name)
masked_data_location = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', masked_data_name)
# mask CT images using labels>0 dilated by 10 pixels
# images and labels are in raw_data_location/'images' and raw_data_location/'labels'
# save masked images in masked_data_location/'images' and original labels in masked_data_location/'labels'

if not os.path.exists(masked_data_location):
    os.makedirs(masked_data_location)
    os.makedirs(os.path.join(masked_data_location, 'images'))
    os.makedirs(os.path.join(masked_data_location, 'labels'))

raw_images = [f for f in os.listdir(os.path.join(raw_data_location, 'images')) if f.endswith('.nii.gz')]
#labels have the same name as images
for img in raw_images:
    img_path = os.path.join(raw_data_location, 'images', img)
    label_path = os.path.join(raw_data_location, 'labels', img)
    img_data = nib.load(img_path).get_fdata()
    img_header, affine = nib.load(img_path).header, nib.load(img_path).affine
    label_data = nib.load(label_path).get_fdata()
    mask = binary_dilation(label_data>0, iterations=10)
    masked_img_data = np.where(mask, img_data, -500)
    masked_img = nib.Nifti1Image(masked_img_data, affine, img_header)
    nib.save(masked_img, os.path.join(masked_data_location, 'images', img))
    # os cp label to masked_data_location
    os.system('cp {} {}'.format(label_path, os.path.join(masked_data_location, 'labels', img)))


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