# a script to dsiplay an black and white png image file

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
plt.switch_backend('TkAgg')



img = mpimg.imread('/Users/mcgoug01/Downloads/DSC_0564_BWsmall.png')
# plot the original image, and 3 altered versions.
#first altered version - inverted intensity
#second altered version - horizontal flip
#third altered version - segmented image where intensity > 0.5 is white, else black

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
print(ax)
ax[0,0].imshow(img[:,:,0], cmap='gray')
ax[0,0].set_title('Original Image')
ax[0,0].axis('off')


ax[0,1].imshow((1-img[:,:,0]).astype(float), cmap='gray')
ax[0,1].set_title('Inverted Image')
ax[0,1].axis('off')

flip = np.clip(img[:,:,0] + np.random.normal(0,0.1,img[:,:,0].shape), 0, 1)

ax[1,0].imshow(flip, cmap='gray')
ax[1,0].set_title('Noised Image')
ax[1,0].axis('off')

ax[1,1].imshow((img[:,:,0] > 0.3).astype(int), cmap='gray')
ax[1,1].set_title('Thresholded Image')
ax[1,1].axis('off')

plt.show()



