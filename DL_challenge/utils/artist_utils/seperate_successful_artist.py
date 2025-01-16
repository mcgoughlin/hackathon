# get list of '.nii.gz' files in first dir and in second dir, and move the files that exist in both dirs to a third dir

import os
import shutil

# get list of '.nii.gz' files in first dir
src_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/tcia_ncct_reg/images'
check_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/masked_tcia_ncct_reg/images'
dst_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/successful_tcia_ncct_reg/images'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

src_files = os.listdir(src_dir)
check_files = os.listdir(check_dir)

# move the files that exist in both dirs to a third dir
for file in src_files:
    if file in check_files:
        shutil.move(os.path.join(src_dir, file), os.path.join(dst_dir, file))
        print(file)