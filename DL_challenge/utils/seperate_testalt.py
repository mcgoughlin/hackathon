# take files from directory A and copy to directory dst if they do not exist in directory B or do directory C
#compare the lowercase version of the file names

import os
import shutil

dir_a = '/Users/mcgoug01/Downloads/test_set_alt/add_ncct_images'
dir_b = '/Users/mcgoug01/Downloads/test_set_alt/add_cect_images'
dir_b2 = '/Users/mcgoug01/Downloads/test_set_alt2/cect_labels'
dir_c = '/Users/mcgoug01/Downloads/test_set_alt2/successfully_coregistered'
dst = '/Users/mcgoug01/Downloads/test_set_alt2/failed_registration'

if not os.path.exists(dst):
    os.makedirs(dst)

files_a = [f for f in os.listdir(dir_a) if f.endswith('.nii.gz')]
files_b = [f for f in os.listdir(dir_b2) if f.endswith('.nii.gz')]
files_c = [f for f in os.listdir(dir_c) if f.endswith('.nii.gz')]
files_b_lower = [f.lower() for f in files_b]
files_c_lower = [f.lower() for f in files_c]

b_counter = 0
c_counter = 0
transferred = 0
for file in files_a:
    if file.lower() in files_b_lower and file.lower() not in files_c_lower:
        shutil.copy(os.path.join(dir_a, file), os.path.join(dst, file))
        print('Copied:', file)
        transferred += 1
    else:
        # print reason for not copying
        if file.lower() not in files_b_lower:
            print('File not in B (not paired):', file)
            b_counter += 1

        if file.lower() in files_c_lower:
            print('File in C (was successful):', file)
            c_counter += 1

print('Files not in B:', b_counter)
print('Files in C:', c_counter)
print('Files transferred:', transferred)