# a script to dsiplay an black and white png image file

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

image = cv2.imread('/Users/mcgoug01/Downloads/DSC_0564_BWsmall.png')
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], np.float32)/16
#create a 5x5 laplacian
laplacian_kernel = np.array([[0, 0, 1, 0, 0],
                            [0, 1, 2, 1, 0],
                            [1, 2, -16, 2, 1],
                            [0, 1, 2, 1, 0],
                            [0, 0, 1, 0, 0]], np.float32)*-1

gaussian_image = cv2.filter2D(image, -1, gaussian_kernel)
log_image = cv2.filter2D(gaussian_image, -1, laplacian_kernel)
diff = log_image - log_image.mean()
sharp = image.copy() + log_image*0.5
#clip sharp to 0, 255
sharp = np.clip(sharp, 0, 255)
sharp = np.array(sharp, np.uint16)

print(image.mean(),image.max(), image.shape)
print(gaussian_image.mean(),gaussian_image.max(), gaussian_image.shape)
print(log_image.mean(),log_image.max(), log_image.shape)
print(sharp.mean(),sharp.max(), sharp.shape)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0,0].imshow(image, cmap='gray')
ax[0,0].set_title('Original Image')
ax[0,0].axis('off')

ax[0,1].imshow(gaussian_image, cmap='gray')
ax[0,1].set_title('Gaussian Smoothed Image')
ax[0,1].axis('off')

ax[1,0].imshow(log_image, cmap='gray')
ax[1,0].set_title('Laplacian of Gaussian')
ax[1,0].axis('off')

ax[1,1].imshow(sharp, cmap='gray')
ax[1,1].set_title('Sharpened Image')
ax[1,1].axis('off')


plt.show()





