# home path contains two subdirs: kidney and cancer segs
# want to create a new seg that contains each of the segs in the two subdirs - kidney is 1, cancer is 2
# resize both maps to 2mm isotropic

import os
import numpy as np
import nibabel as nib
import scipy.ndimage
import pandas as pd

home = '/Users/mcgoug01/Downloads/oxcam/oxcam'
kidney = 'kidney_[2 2 2]mm'
images = 'images'
cancer = 'cancer_[2 2 2]mm'
new_inf = 'new_inferences'
new_img = 'new_images'

kidney_path = os.path.join(home, kidney)
cancer_path = os.path.join(home, cancer)
new_inf_path = os.path.join(home, new_inf)
new_img_path = os.path.join(home, new_img)
image_path = os.path.join(home, images)
if not os.path.exists(new_inf_path):
    os.makedirs(new_inf_path)

if not os.path.exists(new_img_path):
    os.makedirs(new_img_path)

kidney_files = [file for file in os.listdir(kidney_path) if file.endswith('.nii.gz')]
results =[]

for file in kidney_files:
    kid = nib.load(os.path.join(kidney_path, file))
    new_inf_data = kid.get_fdata()
    new_inf_data[new_inf_data > 0] = 1

    canc = nib.load(os.path.join(cancer_path, file))
    canc_data = canc.get_fdata()
    new_inf_data[canc_data > 0] = 2

    spacing = nib.load(os.path.join(kidney_path, file)).header.get_zooms()
    new_spacing = [1,1,1]

    aff = np.eye(4)*new_spacing[0]
    aff[3, 3] = 1
    print(new_inf_data.shape, spacing, new_spacing)
    img = nib.load(os.path.join(image_path, file))
    new_inf_data = scipy.ndimage.zoom(new_inf_data, np.divide(spacing, new_spacing), order=0)
    new_img_data = scipy.ndimage.zoom(img.get_fdata(), np.divide(spacing, new_spacing), order=3)
    print(new_inf_data.shape)
    #dilate cancer seg by 1 voxel
    dilate = np.zeros(new_inf_data.shape, dtype=np.uint8)
    dilate[new_inf_data == 2] = 1
    dilate = scipy.ndimage.binary_dilation(dilate, iterations=1).astype(np.uint8)
    new_inf_data[(dilate == 1) & (new_inf_data==1)] = 2
    # flip image lr
    new_inf_data = np.flip(new_inf_data, axis=0)
    new_img_data = np.flip(new_img_data, axis=0)

    new_inf = nib.Nifti1Image(new_inf_data, aff)
    nib.save(new_inf, os.path.join(new_inf_path, file))
    new_img = nib.Nifti1Image(new_img_data, aff)
    nib.save(new_img, os.path.join(new_img_path, file))

    #find the largest cancer region and extract its segmentation
    labels, n = scipy.ndimage.label(new_inf_data == 2)
    sizes = scipy.ndimage.sum(new_inf_data == 2, labels, range(1, n+1))
    if len(sizes) == 0:
        entry = {'file': file, 'volume': np.nan, 'surface_area': np.nan, 'sphericity': np.nan,
                 'diameter': np.nan, 'exophytic': np.nan}
    else:
        max_label = np.argmax(sizes) + 1
        largest_cancer = np.zeros(new_inf_data.shape, dtype=np.uint8)
        largest_cancer[labels != max_label] = 0
        largest_cancer[labels == max_label] = 1

        #extract volume, surface area, sphericity, largest diameter, and exophytic fraction
        volume = np.sum(largest_cancer)
        # scale volume to mm^3 and then to cm^3
        volume = volume*(new_spacing[0]**3) / 1000
        contour = scipy.ndimage.binary_dilation(largest_cancer, iterations=1).astype(np.uint8) - largest_cancer
        surface_area = np.sum(contour)*2
        # scale surface area to mm^2 and then to cm^2
        surface_area = surface_area*(new_spacing[0]**2) / 100
        sphericity = (np.pi**(1/3))*((6*volume)**(2/3))/surface_area
        diameter = np.max(scipy.ndimage.distance_transform_edt(largest_cancer)*2)
        # scale diameter to mm and then to cm
        diameter = diameter*new_spacing[0] / 10

        #find the exophytic fraction
        #exophytic fraction is ratio of contour that is within new_data==1
        exophytic = 1-np.sum(contour[new_inf_data==1])/np.sum(contour)
        entry = {'file': file, 'volume': volume, 'surface_area': surface_area, 'sphericity': sphericity, 'diameter': diameter, 'exophytic': exophytic}
    results.append(entry)
    print(entry)

df = pd.DataFrame(results)
df.to_csv(os.path.join(home, 'results.csv'), index=False)
print(df.head())