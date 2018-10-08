# Public libraries
import os
import numpy as np
import cv2

# Private libraries
from disparity_ssd import disparity_ssd

if 1 == 0:
	## 1-a
	# Read images
	L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
	R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)
	rows, columns = L.shape
	# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
	D_L = disparity_ssd(L, R)
	D_R = disparity_ssd(R, L)
	# Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-1-a-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-1-a-2.png', D_R_equ)

	## 2-a
	# Read images
	L = cv2.imread('input/pair1-L.png', 0)
	R = cv2.imread('input/pair1-R.png', 0)
	rows, columns = L.shape
	# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
	D_L = disparity_ssd(L, R)
	D_R = disparity_ssd(R, L)
	# Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-2-a-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-2-a-2.png', D_R_equ)

## 3-a
# Read images
L = cv2.imread('input/pair1-L.png', 0)
R = cv2.imread('input/pair1-R.png', 0)
rows, columns = L.shape
# Add gaussian noise to L
mu = 0
sigma = 3
noise = np.random.normal(mu, sigma, rows*columns)
L = noise.reshape(rows, columns) + L
cv2.imwrite('output/noisy.png', L)
# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
D_L = disparity_ssd(L, R)
D_R = disparity_ssd(R, L)
# Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# Note: They may need to be scaled/shifted before saving to show results properly
D_L_equ = cv2.equalizeHist(D_L)
cv2.imwrite('output/ps2-3-a-1.png', D_L_equ)
D_R_equ = cv2.equalizeHist(D_R)
cv2.imwrite('output/ps2-3-a-2.png', D_R_equ)

## 3-b
# Read images
L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)
rows, columns = L.shape
# Add brightness to L
L = L * 1.1 # 10%
cv2.imwrite('output/bright.png', L)
# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
D_L = disparity_ssd(L, R)
D_R = disparity_ssd(R, L)
# Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# Note: They may need to be scaled/shifted before saving to show results properly
D_L_equ = cv2.equalizeHist(D_L)
cv2.imwrite('output/ps2-3-b-1.png', D_L_equ)
D_R_equ = cv2.equalizeHist(D_R)
cv2.imwrite('output/ps2-3-b-2.png', D_R_equ)
