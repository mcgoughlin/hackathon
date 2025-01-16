import os

path = '/rds/project/rds-sDuALntK11g/IMAGING_nifti'
save_path = '/rds/project/rds-sDuALntK11g/IMAGING_niftipairs'

#copy the nifti files to a new directory
# src files have the following subdir schema:
# /rds/project/rds-sDuALntK11g/IMAGING_nifti
#   -/subject_id/
#       -/0/
#           -/ce.nii.gz
#           -/nc.nii.gz
#       -/1/
#           -/ce.nii.gz
#           -/nc.nii.gz

# dest files will have the following subdir schema:
# /rds/project/rds-sDuALntK11g/IMAGING_niftipairs
#   -/cect/
#       -/images/
#           -/subject_id_0.nii.gz
#           -/subject_id_1.nii.gz
#   -/ncct/
#       -/images/
#           -/subject_id_0.nii.gz
#           -/subject_id_1.nii.gz

# create the directories
os.makedirs(os.path.join(save_path, 'ce', 'images'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'nc', 'images'), exist_ok=True)

modalities = ['ce.nii.gz','nc.nii.gz']

# copy the files
for subject_id in os.listdir(path):
    for study in [subdir for subdir in os.listdir(os.path.join(path, subject_id)) if os.path.isdir(os.path.join(path, subject_id, subdir))]:
        #check that the study has both modalities
        if not all([m in os.listdir(os.path.join(path, subject_id, study)) for m in modalities]):
            print(f'{subject_id} {study} does not have both modalities')
            continue
        for modality in modalities:
            src_file = os.path.join(path, subject_id, study, modality)
            dest_file = os.path.join(save_path, modality[:-7], 'images', f'{subject_id}_{study}.nii.gz')
            os.system(f'cp {src_file} {dest_file}')

