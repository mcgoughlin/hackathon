import os
os.environ['OV_DATA_BASE'] = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/'
from KCD.Segmentation.Inference.endtoend import EnsembleSeg, masking
import numpy as np
from skimage.measure import label
import nibabel as nib

#record time taken to run each function in the script
import time


def detection(home, pred_path, op_point,dataset, name):
    assert os.path.exists(home), 'Path does not exist'
    assert os.path.exists(pred_path), 'Path does not exist'

    predictions = [x for x in os.listdir(pred_path) if 'nii' in x]
    save_path = os.path.join(home, 'detections',dataset+'_ROCv3', str(name))

    save_path_pos = os.path.join(save_path, 'positives_nii')
    save_path_neg = os.path.join(save_path, 'negatives_nii')

    pos_csv = os.path.join(save_path, 'positive.csv')
    neg_csv = os.path.join(save_path, 'negative.csv')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_pos):
        os.makedirs(save_path_pos)
    if not os.path.exists(save_path_neg):
        os.makedirs(save_path_neg)


    #if pos or neg csv already exists, delete it
    if os.path.exists(pos_csv):
        os.remove(pos_csv)
    if os.path.exists(neg_csv):
        os.remove(neg_csv)

    #open pos and neg csv files
    pos_csv = open(pos_csv, 'w')
    neg_csv = open(neg_csv, 'w')


    for pred_fn in predictions:
        print(pred_fn)
        pred_path_case = os.path.join(pred_path, pred_fn)
        pred_nib = nib.load(pred_path_case)
        pred = pred_nib.get_fdata()
        thresholded_pred = np.zeros_like(pred)
        voxvol = np.prod(pred_nib.header.get_zooms())

        pred_foreground = (pred > op_point).astype(int)
        pred_components = label(np.squeeze(pred_foreground))

        # if any component is greater than 500mm^3, append to pos_csv the name of the file,
        # the number of components, and the volume of the components. Otherwise, append to neg_csv
        component_volumes = [(pred_components == i).sum() *voxvol for i in range(1, np.max(pred_components) + 1)]

        if any([x > 500 for x in component_volumes]):
            pos_csv.write(pred_fn + ',' + str(max(component_volumes)) + '\n')
        else:
            neg_csv.write(pred_fn + ',' + str(0) + '\n')

        # insert 1 wherever a component exists that has a volume greater than 500mm^3
        for i in range(1, np.max(pred_components) + 1):
            if component_volumes[i-1] > 500:
                thresholded_pred[pred_components == i] = 1

        # if thresholded_pred has any 1s, save to save_path_pos, else save to save_path_neg
        thresholded_nib = nib.Nifti1Image(thresholded_pred, pred_nib.affine, header = pred_nib.header)
        if np.any(thresholded_pred):
            nib.save(thresholded_nib, os.path.join(save_path_pos, pred_fn))
        else:
            nib.save(thresholded_nib, os.path.join(save_path_neg, pred_fn))

    pos_csv.close()
    neg_csv.close()



def EndToEnd(dataset, stage1_path, stage_2path, op_points = [0.001,0.0089,0.01,0.02,0.03,0.04,0.05,0.1,0.150,0.2,0.3,0.365,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99,0.999],
             op_point_names = [0.001,'high_sens',0.01,0.02,0.03,0.04,0.05,0.1,'high_acc',0.2,0.3,'high_spec',0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99,0.999],cont_multiplier=1e5):

    print('Running EndToEnd')
    print('\n' * 5)
    #print time and many spaces after
    start = time.time()
    print(f'Begin at {time.ctime()}')
    print('\n'*5)

    s1 = EnsembleSeg.Ensemble_Seg(dataset,name='v3s1',
                            seg_fp=stage1_path,
                            spacing=np.array([4] * 3),
                            do_prep=True, do_infer=True,
                            is_cect=True, cont=False,overlap=0.5,
                            batch_size=16, patch_size=64)

    print('Finished first stage segmentation at', time.ctime())
    print('\n'*5)


    dataset_path = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', dataset, 'images')
    masked_dataset_path = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'v3masked_' + dataset, 'images')
    pred_path = os.path.join(os.environ['OV_DATA_BASE'], 'predictions_nii', dataset, 'v3s1_[4 4 4]mm')

    masking.mask_dataset(dataset_path, pred_path, masked_dataset_path)

    print('Finished masking dataset at', time.ctime())

    s2 = EnsembleSeg.Ensemble_Seg('v3masked_' + dataset,name='v3s2',
                            seg_fp=stage_2path,
                            spacing=np.array([2] * 3),
                            do_prep=True, do_infer=True,
                            is_cect=True, cont=True,overlap=0.5,
                            batch_size=16, patch_size=64, cont_multiplier=cont_multiplier)

    print('Finished second stage segmentation at', time.ctime())

    cont_pred_path = os.path.join(os.environ['OV_DATA_BASE'], 'predictions_nii', 'v3masked_' + dataset, 'v3s2_[2 2 2]mm_cont')

    for i, op_point in enumerate(op_points):
        detection(os.environ['OV_DATA_BASE'], cont_pred_path, op_point*cont_multiplier,dataset, op_point_names[i])

    end = time.time()
    print('Finished detection at', time.ctime())
    print('\n'*5)

    # print time taken to run the script, and time per cases (there are 4001 cases)
    print(f'Time taken to run script: {end-start}s')
    print('\n'*5)



if __name__ == '__main__':
    s1_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/coreg_cect_v3/4mm_kidneybinary_customPP/6_finetune_trained'
    s2_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/masked_coreg_cect_v3/2mm_cancerbinary_customPP/6_finetune_trained'

    EndToEnd('test_set_cect', s1_path, s2_path)