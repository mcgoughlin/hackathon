import zipfile
import os

src_dir = '/rds/project/rds-sDuALntK11g/IMAGING'
dst_dir = '/rds/project/rds-sDuALntK11g/IMAGING_extracted'

src_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.zip')]
for src_file in src_files:
    with zipfile.ZipFile(src_file, 'r') as zip_ref:
        zip_ref.extractall(dst_dir)
