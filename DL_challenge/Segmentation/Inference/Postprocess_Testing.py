import os
import numpy as np
from skimage.measure import label
import nibabel as nib

#record time taken to run each function in the script
import time


def pp(im_path, pred_path, save_path):
    assert os.path.exists(im_path), 'Path does not exist'
    assert os.path.exists(pred_path), 'Path does not exist'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    predictions = [x for x in os.listdir(pred_path) if 'nii' in x]

    pos_csv = os.path.join(save_path, 'positive_pp.csv')

    if os.path.exists(pos_csv):
        os.remove(pos_csv)

    pos_csv = open(pos_csv, 'w')

    for pred_fn in predictions:
        pred_path_case = os.path.join(pred_path, pred_fn)
        im_path_case = os.path.join(im_path, pred_fn)
        pred_nib = nib.load(pred_path_case)
        im_nib = nib.load(im_path_case)
        pred = pred_nib.get_fdata()
        im = im_nib.get_fdata()

        pred_foreground = (pred > 0).astype(int)
        pred_components = label(np.squeeze(pred_foreground))

        # check the average intensity of img within the component - if less than 20 or more than 70, discard
        # this component

        # after this, if there remains any positive component, save the image

        for i in range(1, np.max(pred_components) + 1):
            component = (pred_components == i)
            if np.mean(im[component]) < 20 or np.mean(im[component]) > 70:
                pred_components[component] = 0

        pred_components = label(np.squeeze(pred_components))

        if np.max(pred_components) > 0:
            save_path_case = os.path.join(save_path, pred_fn)
            nib.save(nib.Nifti1Image(pred_components, pred_nib.affine,header = pred_nib.header), save_path_case)
            pos_csv.write(pred_fn + ',' + str(np.max(pred_components)) + '\n')

    pos_csv.close()



if __name__ == '__main__':
    im_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/ykst/images'
    pred_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/detections/ykst/high_acc/positives_nii'
    save_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/detections/ykst/high_acc/positives_pp_nii'
    pp(im_path, pred_path, save_path)