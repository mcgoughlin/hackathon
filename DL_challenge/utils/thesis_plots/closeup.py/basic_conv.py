# a script to dsiplay an black and white png image file

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

image = cv2.imread('/Users/mcgoug01/Downloads/DSC_0564_BWsmall.png')

h_kernel = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], np.float32)

h_image = np.abs(cv2.filter2D(image, -1, h_kernel))

v_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], np.float32)

v_image = np.abs(cv2.filter2D(image, -1, v_kernel))
combined =(h_image+v_image)/2
combined= combined-np.min(combined)
combined= combined/np.max(combined)
# convert combined into f32
combined = np.array(combined, np.float32)
print(combined.shape)

#print max and min of h_image and v_image and image
print(f'Max of image: {np.max(image)}')
print(f'Min of image: {np.min(image)}')
print(f'Max of h_image: {np.max(h_image)}')
print(f'Min of h_image: {np.min(h_image)}')
print(f'Max of v_image: {np.max(v_image)}')
print(f'Min of v_image: {np.min(v_image)}')
print(f'Max of combined: {np.max(combined)}')
print(f'Min of combined: {np.min(combined)}')


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0,0].imshow(image, cmap='gray')
ax[0,0].set_title('Original Image')
ax[0,0].axis('off')

ax[0,1].imshow(h_image, cmap='gray')
ax[0,1].set_title('Horizontal Edges')
ax[0,1].axis('off')

ax[1,0].imshow(v_image, cmap='gray')
ax[1,0].set_title('Vertical Edges')
ax[1,0].axis('off')

ax[1,1].imshow(combined, cmap='gray')
ax[1,1].set_title('Edge Magnitude')
ax[1,1].axis('off')

plt.show()





