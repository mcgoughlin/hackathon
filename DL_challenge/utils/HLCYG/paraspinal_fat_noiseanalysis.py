import sys

import nibabel as nib
import os
import numpy as np
from skimage.measure import regionprops

def get_nii_image(image_path):
    '''
    :param image_path: str
    :return hu_array: arr
    '''
    image = nib.load(image_path).get_fdata()
    aff = nib.load(image_path).affine
    return image, aff, nib.load(image_path).header

if __name__ == '__main__':
    original_image_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/coreg_ncct_v3/images'
    original_label_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/coreg_ncct_v3/labels'
    doses = [0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    preformatted_dose_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/coreg_ncct_v3/noised_ims_'
    save_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/coreg_ncct_v3/noise_analysis.csv'

    #open a csv file to write the results
    with open(save_path, 'w') as f:
        f.write('file,1.0')
        for dose in doses:
            f.write(f',{dose}')
        f.write('\n')
    results = []
    for file in [f for f in os.listdir(original_image_path) if f.endswith('.nii.gz')]:
        datum = {'file': file}
        orig_im_nib = os.path.join(original_image_path, file)
        orig_lb_nib = os.path.join(original_label_path, file)
        orig_im, aff, header = get_nii_image(orig_im_nib)
        orig_lb, _, _ = get_nii_image(orig_lb_nib)

        orig_lb = (orig_lb>0)
        # find the for the whole foreground of the label - regionprops
        bbox = regionprops(orig_lb.astype(np.uint8))[0].bbox

        # find all pixels in the image that are within the bounding box and between -150 and -50 HU
        # and create a mask of the same shape as the image
        mask = np.zeros_like(orig_im)
        mask[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] = 1
        print(bbox)
        mask = (orig_im > -150) & (orig_im < -50) & (mask>0)
        # numbers from this paper: https://www.tandfonline.com/doi/full/10.1080/0284186X.2020.1800087#abstract

        #print number of pixels in mask
        # print(np.sum(mask))
        #
        # # print number of pixels that match each condition
        # print(np.sum(orig_im > -150))
        # print(np.sum(orig_im < -50))
        # print(np.sum(mask>1))
        # print('######')
        # # now each 2nd order combination
        # print(np.sum((orig_im > -150) & (orig_im < -50)))
        # print(np.sum((orig_im > -150) & (mask>1)))
        # print(np.sum((orig_im < -50) & (mask>1)))
        # print('@@@@@@')
        #


        # find std dev in original image
        orig_im_std = np.std(orig_im[mask])
        print(orig_im_std)
        sys.stdout.flush()
        datum['1.0'] = orig_im_std
        for dose in doses:
            dose_path = preformatted_dose_path + str(dose)
            if not os.path.exists(dose_path):
                # raise an error if the path does not exist
                raise FileNotFoundError(f'Path not found: {dose_path}')

            noised_im_nib = os.path.join(dose_path, file)
            noised_im, _, _ = get_nii_image(noised_im_nib)
            noised_im_std = np.std(noised_im[mask])
            datum[str(dose)] = noised_im_std
            print(datum)
        results.append(datum)
        print(f'Finished {file}')
        print(datum)
        with open(save_path, 'a') as f:
            f.write(f"{datum['file']},{datum['1.0']}")
            for dose in doses:
                f.write(f",{datum[str(dose)]}")
            f.write('\n')
        print()
        sys.stdout.flush()

    print('Done!')

