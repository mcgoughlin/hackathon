import nibabel as nib
import numpy as np
import os
import torch
from scipy.ndimage.morphology import binary_fill_holes
import scipy.ndimage as spim
from skimage.measure import regionprops
from KCD.Detection.Preprocessing.ObjectFiles import object_creation_utils as ocu
from skimage.segmentation import watershed

# we wan to find the optimum confidence threshold for determining the presence of cancerous voxel labels,
# and the optimum size threshold for determining the presence of a cancerous region
# we will do this by finding the threshold that maximises the dice score on a dataset

cancer_infp = os.path.join('/Users/mcgoug01/Downloads/test_results/nii_pred_flip')
kidney_infp = os.path.join('/Users/mcgoug01/Downloads/test_results/mask_flip')
cancer_label = os.path.join('/Users/mcgoug01/Downloads/test_results/nii_lb')
confidence_thresholds = np.append(np.arange(0,0.1,0.02),np.arange(0.1, 0.9, 0.1))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.9,0.98,0.01))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.98,1.005,0.005))

# confidence_thresholds = [0.5]
print(confidence_thresholds)
vol = 400

results =[]

#index 0 is left kidney, index 1 is right kidney
test_labels = {'20.nii.gz':[0,0],
                '74.nii.gz':[0,0],
                '131.nii.gz':[0,0],
                '132.nii.gz':[0,0],
                '133.nii.gz':[0,0],
                '135.nii.gz':[0,0],
                '136.nii.gz':[0,0],
                '137.nii.gz':[0,0],
                '138.nii.gz':[0,0],
                '139.nii.gz':[0,0],
                '189.nii.gz':[0,0],
                '191.nii.gz':[0,0],
                '396.nii.gz':[0,0],
                '397.nii.gz':[0,0],
                '398.nii.gz':[0,0],
                '767.nii.gz':[0,0],
                '772.nii.gz':[0,0],
                'Rcc_002.nii.gz':[0,1],
                'Rcc_005.nii.gz':[1,0],
                'Rcc_009.nii.gz':[0,1],
                'Rcc_010.nii.gz':[1,0],
                'Rcc_012.nii.gz':[1,0],
                'Rcc_018.nii.gz':[0,1],
                'Rcc_021.nii.gz':[0,1],
                'Rcc_022.nii.gz':[0,1],
                'Rcc_024.nii.gz':[0,1],
                'Rcc_026.nii.gz':[0,1],
                'Rcc_029.nii.gz':[1,0],
                'Rcc_048.nii.gz':[1,0],
                'Rcc_056.nii.gz':[1,0],
                'Rcc_063.nii.gz':[1,0],
                'Rcc_065.nii.gz':[1,0],
                'Rcc_070.nii.gz':[1,0],
                'Rcc_073.nii.gz':[1,0],
                'Rcc_077.nii.gz':[0,1],
                'Rcc_079.nii.gz':[0,1],
                'Rcc_080.nii.gz':[1,0],
                'Rcc_086.nii.gz':[0,1],
                'Rcc_091.nii.gz':[1,0],
                'Rcc_092.nii.gz':[1,0],
                'Rcc_094.nii.gz':[0,1],
                'Rcc_097.nii.gz':[1,0],
                'Rcc_098.nii.gz':[0,1],
                'Rcc_105.nii.gz':[1,0],
                'Rcc_106.nii.gz':[0,1],
                'Rcc_109.nii.gz':[1,0],
                'Rcc_110.nii.gz':[1,0],
                'Rcc_112.nii.gz':[1,0],
                'Rcc_119.nii.gz':[1,0],
                'Rcc_130.nii.gz':[0,1],
                'Rcc_133.nii.gz':[1,0],
                'Rcc_135.nii.gz':[0,1],
                'Rcc_137.nii.gz':[0,1],
                'Rcc_139.nii.gz':[0,1],
                'Rcc_159.nii.gz':[1,0],
                'Rcc_163.nii.gz':[1,0],
                'Rcc_165.nii.gz':[1,0],
                'Rcc_169.nii.gz':[0,0],
                'Rcc_175.nii.gz':[0,1],
                'Rcc_184.nii.gz':[0,1],
                'Rcc_187.nii.gz':[0,1],
                'Rcc_191.nii.gz':[1,0],
                'Rcc_196.nii.gz':[1,0],
                'Rcc_202.nii.gz':[0,1]}


for file in [path for path in os.listdir(cancer_infp) if path.endswith('.nii.gz')]:
    print(file)
    if file == 'Rcc_036.nii.gz': continue
    if file == 'RCC_036.nii.gz': continue
    cancer_inf = nib.load(os.path.join(cancer_infp, file))
    kidney_inf = nib.load(os.path.join(kidney_infp, file))
    cancer_lb = nib.load(os.path.join(cancer_label, 'Rcc' + file[3:]))

    left_label, right_label = test_labels['Rcc' + file[3:]]

    # print(f'Processing {file} - left kidney: {left_label}, right kidney: {right_label}')
    # #find spacing
    spacing = cancer_inf.header['pixdim'][1:4]

    # resize all to 2mm spacing using torch
    cancer_inf = torch.from_numpy(cancer_inf.get_fdata()).unsqueeze(0).unsqueeze(0) / 1000
    kidney_inf = torch.from_numpy(kidney_inf.get_fdata()).unsqueeze(0).unsqueeze(0)
    cancer_lb = torch.from_numpy(cancer_lb.get_fdata()).unsqueeze(0).unsqueeze(0)
    cancer_inf = torch.nn.functional.interpolate(cancer_inf, scale_factor=tuple(spacing / 2),
                                                 mode='trilinear').squeeze()
    kidney_inf = torch.nn.functional.interpolate(kidney_inf, scale_factor=tuple(spacing / 2), mode='nearest').squeeze()
    cancer_lb = torch.nn.functional.interpolate(cancer_lb, scale_factor=tuple(spacing / 2), mode='nearest').squeeze()

    kidney_inf = kidney_inf.numpy()
    kidney_inf = spim.binary_dilation(kidney_inf, iterations=10)
    kidney_inf = torch.from_numpy(kidney_inf)
    # print(kidney_inf.shape)

    size = (cancer_lb==1).sum()

    left_size, right_size = size*left_label, size*right_label
    # if largest cancer region in cancer_lb is less than 33510mm cubed, or 4189 voxels, than small_cancer = True
    # all none addenbrookes cases should be left == right == 0, i.e. no cancer.
    if right_label==0 and left_label==0: small_cancer = True
    elif not file.startswith('RCC'): assert False
    elif size < 4189: small_cancer = True

    #find cancerous voxels
    for conf in confidence_thresholds:
        cancer_voxels = cancer_inf > conf
        # fill holes in cancerous voxels
        cancer_voxels = binary_fill_holes(cancer_voxels)
        #filter out regions with less than vol voxels using regionprops and spim
        cancer_voxels = spim.label(cancer_voxels)[0]
        cancer_regions = np.array([region for region in regionprops(cancer_voxels) if region.area > vol])
        # print(f'Number of regions in {file}: {len(cancer_regions)}')

        # reconstruct cancerous voxels from regions over vol
        cancer_voxels = np.zeros_like(cancer_voxels)
        for region in cancer_regions:
            cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]] = 2

        # loop through all cancerous labels - checking if true or false positive
        positions = ['right' if region.centroid[0]>kidney_inf.shape[0]//2 else 'left' for region in regionprops(spim.label(kidney_inf == 1)[0])]
        ordered_positions = [left_label if pos == 'left' else right_label for pos in positions]
        ordered_size = [left_size if pos == 'left' else right_size for pos in positions]

        for pos,lb,size,region in zip(positions,ordered_positions,ordered_size,regionprops(spim.label(kidney_inf == 1)[0])):
            prediction_region = cancer_voxels[region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]]
            # print(f'Position: {pos}, Label: {lb}, Size: {size}, Confidence: {conf}, Prediction: {prediction_region.max()}')
            if prediction_region.max() ==2 :
                entry = {'file':file,'confidence':conf,'position':pos, 'label':lb,'size_mm':size.item()*8}
                results.append(entry)

            # # if no entries appended for a case and position, append a 'zero' conf entry
            # if len([entry for entry in results if entry['file']==file and entry['position']==pos])==0:
            #     entry = {'file':file,'confidence':0,'position':pos, 'label':lb,'size_mm':size*8}
            #     results.append(entry)


import pandas
# group by file, position, label - find max confidence for each
df = pandas.DataFrame(results)
print(df)
df = df.groupby(['file','position','label','size_mm']).max().reset_index()
df.to_csv(os.path.join('/Users/mcgoug01/Downloads/test_results', 'casewise_confidences.csv'))