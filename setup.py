from setuptools import setup,find_packages

setup(
    name='KCD_early',
    packages=find_packages('KCD_src', exclude=['test']),
    install_requires=['torch','matplotlib','nibabel','dicom2nifti',
              'numpy','opencv-python','pandas','pydicom',
              'scikit-image','scikit-learn','scipy','numpy-stl',
              'simpleitk','torchvision','rt_utils'],
    python_requires='>=3.10',
    description='Python Package for the early detection of renal cancer',
    version='0.1',
    url='https://github.com/mcgoughlin/hackathon',
    author='bmcgough',
    author_email='billy.mcgough1@hotmail.com',
    keywords=['pip','pytorch','cancer']
    )