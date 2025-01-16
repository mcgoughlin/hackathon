import os
import shutil
import numpy as np

#the job of this script is to search for all the subfolders in the home directory that
#contain the substring '_HLCYG_'

#each of these folders will have a subfolder called '2mm_cancerbinary_customPP', and in that subfolder we have further subfolders 'images', 'labels', 'fingerprints',
#. 2mm_cancerbinary_customPP will also have the files 'preprocessing_parameters.pkl' and 'preprocessing_parameters.txt'

#this script create a new folder, replacing the substring '_HLCYG_' with '_HLCYGnoiseonly_'
#and copies the 'images', 'labels', 'fingerprints', 'preprocessing_parameters.pkl' and 'preprocessing_parameters.txt'
#folders/files to the new folder

# only the files with the substring '_noised' in their name will be copied to the
#new folder

#the new folder will be created in the same directory as the original folder

home_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed'

#find all the subfolders in the home directory that contain the substring '_HLCYG_'
subfolders = []
for root, dirs, files in os.walk(home_dir):
    for dir in dirs:
        if '_HLCYG_' in dir and dir.startswith('AImasked'):
            subfolders.append(os.path.join(root, dir))
print(subfolders)
#iterate through the subfolders
for subfolder in subfolders:
    #find the subfolder '2mm_cancerbinary_customPP'
    for root, dirs, files in os.walk(subfolder):
        for dir in dirs:
            if dir == '2mm_cancerbinary_customPP':
                #create the new folder
                new_folder = subfolder.replace('_HLCYG_', '_HLCYGnoiseonly_')
                new_folder = os.path.join(new_folder, '2mm_cancerbinary_customPP')
                print(new_folder)
                os.makedirs(new_folder, exist_ok=True)

                #make the folders the 'images', 'labels', 'fingerprints',
                # and copy the files'preprocessing_parameters.pkl' and 'preprocessing_parameters.txt'
                #to the new folder
                os.makedirs(os.path.join(new_folder, 'images'), exist_ok=True)
                os.makedirs(os.path.join(new_folder, 'labels'), exist_ok=True)
                os.makedirs(os.path.join(new_folder, 'fingerprints'), exist_ok=True)
                shutil.copy(os.path.join(subfolder, '2mm_cancerbinary_customPP', 'preprocessing_parameters.pkl'), os.path.join(new_folder, 'preprocessing_parameters.pkl'))
                shutil.copy(os.path.join(subfolder, '2mm_cancerbinary_customPP', 'preprocessing_parameters.txt'), os.path.join(new_folder, 'preprocessing_parameters.txt'))

                #copy the files with the substring '_noised' in their name within 'images', 'labels', 'fingerprints'
                # to the new folders
                for root, dirs, files in os.walk(os.path.join(subfolder, '2mm_cancerbinary_customPP', 'images')):
                    for file in files:
                        if '_noised' in file:
                            shutil.copy(os.path.join(subfolder, '2mm_cancerbinary_customPP', 'images', file), os.path.join(new_folder, 'images', file))
                            shutil.copy(os.path.join(subfolder, '2mm_cancerbinary_customPP', 'labels', file), os.path.join(new_folder, 'labels', file))
                            shutil.copy(os.path.join(subfolder, '2mm_cancerbinary_customPP', 'fingerprints', file), os.path.join(new_folder, 'fingerprints', file))

            #find the subfolder '4mm_kidneybinary_customPP'
            elif dir == '4mm_kidneybinary_customPP':
                #create the new folder
                new_folder = subfolder.replace('_HLCYG_', '_HLCYGnoiseonly_')
                new_folder = os.path.join(new_folder, '4mm_kidneybinary_customPP')
                print(new_folder)
                os.makedirs(new_folder, exist_ok=True)

                #make the folders the 'images', 'labels', 'fingerprints',
                # and copy the files'preprocessing_parameters.pkl' and 'preprocessing_parameters.txt'
                #to the new folder
                os.makedirs(os.path.join(new_folder, 'images'), exist_ok=True)
                os.makedirs(os.path.join(new_folder, 'labels'), exist_ok=True)
                os.makedirs(os.path.join(new_folder, 'fingerprints'), exist_ok=True)
                shutil.copy(os.path.join(subfolder, '4mm_kidneybinary_customPP', 'preprocessing_parameters.pkl'), os.path.join(new_folder, 'preprocessing_parameters.pkl'))
                shutil.copy(os.path.join(subfolder, '4mm_kidneybinary_customPP', 'preprocessing_parameters.txt'), os.path.join(new_folder, 'preprocessing_parameters.txt'))

                #copy the files with the substring '_noised' in their name within 'images', 'labels', 'fingerprints'
                # to the new folders
                for root, dirs, files in os.walk(os.path.join(subfolder, '4mm_kidneybinary_customPP', 'images')):
                    for file in files:
                        if '_noised' in file:
                            shutil.copy(os.path.join(subfolder, '4mm_kidneybinary_customPP', 'images', file), os.path.join(new_folder, 'images', file))
                            shutil.copy(os.path.join(subfolder, '4mm_kidneybinary_customPP', 'labels', file), os.path.join(new_folder, 'labels', file))
                            shutil.copy(os.path.join(subfolder, '4mm_kidneybinary_customPP', 'fingerprints', file), os.path.join(new_folder, 'fingerprints', file))
