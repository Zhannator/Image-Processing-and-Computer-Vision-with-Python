# ps2
import os
import numpy as np
import cv2

from disparity_ssd import disparity_ssd

## 1-a
# Read images
L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)

# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
D_L = disparity_ssd(L, R)
cv2.imwrite('output/ps2-1-a-1.png', D_L)
D_R = disparity_ssd(R, L)
cv2.imwrite('output/ps2-1-a-2.png', D_R)

# TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# Note: They may need to be scaled/shifted before saving to show results properly


# TODO: Rest of your code here
