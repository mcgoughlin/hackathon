# a script to dsiplay an black and white png image file

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
plt.switch_backend('TkAgg')



img = mpimg.imread('/Users/mcgoug01/Downloads/DSC_0564_BWsmall.png')
fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.5, 1]}, figsize = (15,5))
# highlight a square in the image with a red box from (72,40), to (98,66)
a0.imshow(img, cmap = 'gray')
a0.plot([84,98],[36,36], color = 'r', linewidth = 2)
a0.plot([84,84],[36,50], color = 'r', linewidth = 2)
a0.plot([98,98],[36,50], color = 'r', linewidth = 2)
a0.plot([84,98],[50,50], color = 'r', linewidth = 2)
a0.set_aspect('auto')
a0.axis('off')
# extract the square and plot it in the second plot
square = img[36:50,84:98]

a1.imshow(square, cmap = 'gray')
# write the pixel values in the square
for i in range(14):
    for j in range(14):
        print(square[i,j,0])
        a1.text(j,i, str(square[i,j,0])[:3], ha = 'center', va = 'center', color = 'r', fontsize = 9)
a1.axis('off')
plt.show()


