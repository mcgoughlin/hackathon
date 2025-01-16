# a script to dsiplay an black and white png image file

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
plt.switch_backend('TkAgg')



img = mpimg.imread('/Users/mcgoug01/Downloads/DSC_0564_BWsmall.png')
# plot the original image, and 3 altered versions.
#first altered version - inverted intensity
#second altered version - translation by 5,10
#third altered version - shear imagee by 20 degrees

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
print(ax)
ax[0,0].imshow(img[:,:,0], cmap='gray')
ax[0,0].set_title('Original Image')
ax[0,0].axis('off')

# Rotate image by 30 degrees and plot
from scipy.ndimage import rotate
rotated_img = rotate(img[:,:,0], 30, reshape=False)
ax[0,1].imshow(rotated_img, cmap='gray')
ax[0,1].set_title('Rotated Image')
ax[0,1].axis('off')

# Translate image by 5,10 and plot
from scipy.ndimage import shift
translated_img = shift(img[:,:,0], (10,40))
ax[1,0].imshow(translated_img, cmap='gray')
ax[1,0].set_title('Translated Image')
ax[1,0].axis('off')

# Shear image by 20 degrees and plot
from scipy.ndimage import affine_transform
def shear(input, shear):
    shear1, shear2 = shear
    def shear_matrix(shear1, shear2):
        return np.array([[1, shear1], [shear2, 1]])
    return affine_transform(input, shear_matrix(shear1,shear2))

sheared_img = shear(img[:,:,0], (0.05,0.2))
ax[1,1].imshow(sheared_img, cmap='gray')
ax[1,1].set_title('Sheared Image')
ax[1,1].axis('off')

plt.show()



