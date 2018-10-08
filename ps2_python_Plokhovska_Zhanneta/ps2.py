# Public libraries
import os
import numpy as np
import cv2
import warnings

# Private libraries
from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr

def main():
	
	## 1-a: Basic stereo algorithm on simple grayscale image
	print "-----------------------1-A-----------------------"
	# Read images
	L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
	R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)
	rows, columns = L.shape
	# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
	D_L = disparity_ssd(L, R)
	D_R = disparity_ssd(R, L)
	# Save output images
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-1-a-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-1-a-2.png', D_R_equ)

	## 2-a: Basic stereo algorithm on real image
	print "-----------------------2-A-----------------------"
	# Read images
	L = cv2.imread('input/pair1-L.png', 0)
	R = cv2.imread('input/pair1-R.png', 0)
	rows, columns = L.shape
	# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
	D_L = disparity_ssd(L, R)
	D_R = disparity_ssd(R, L)
	# Save output images
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-2-a-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-2-a-2.png', D_R_equ)

	## 3-a: Basic stereo algorithm on real image with added noise
	print "-----------------------3-A-----------------------"
	# Add gaussian noise to L
	mu = 0
	sigma = 15
	noise = np.random.normal(mu, sigma, rows*columns)
	L_noise = noise.reshape(rows, columns) + L
	# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
	D_L = disparity_ssd(L_noise, R)
	D_R = disparity_ssd(R, L_noise)
	# Save output images
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-3-a-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-3-a-2.png', D_R_equ)

	## 3-b: Basic stereo algorithm on real image with added brightness
	print "-----------------------3-B-----------------------"
	# Add brightness to L
	L_bright = L * 1.1 # 10%
	# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
	D_L = disparity_ssd(L_bright, R)
	D_R = disparity_ssd(R, L_bright)
	# Save output images
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-3-b-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-3-b-2.png', D_R_equ)

	## 4-a: Normalized correlation on normal image
	print "-----------------------4-A-----------------------"
	# Compute disparity (using method disparity_ncorr defined in disparity_ncorr.py)
	D_L = disparity_ncorr(L, R)
	D_R = disparity_ncorr(R, L)
	# Save output images
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-4-a-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-4-a-2.png', D_R_equ)
	
	## 4-b: Normalized correlation on real image with added noise/brightness
	print "-----------------------4-B-----------------------"	
	# Compute disparity (using method disparity_ncorr defined in disparity_ncorr.py)
	D_L = disparity_ncorr(L_noise, R)
	D_R = disparity_ncorr(R, L_noise)
	# Save output images
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-4-b-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-4-b-2.png', D_R_equ)	
	# Compute disparity (using method disparity_ncorr defined in disparity_ncorr.py)
	D_L = disparity_ncorr(L_bright, R)
	D_R = disparity_ncorr(R, L_bright)
	# Save output images
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-4-b-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-4-b-2.png', D_R_equ)
	
	## 5-a: Normalized correlation on smooth/sharp/etc. images
	print "-----------------------5-A-----------------------"
	# Read images
	L = cv2.imread('input/pair2-L.png', 0)
	R = cv2.imread('input/pair2-R.png', 0)
	rows, columns = L.shape
	# Smooth - GaussianBlur
	L_smooth = (cv2.GaussianBlur(L.copy(), (3, 3), 3)).astype('uint8')
	R_smooth = (cv2.GaussianBlur(R.copy(), (3, 3), 3)).astype('uint8')
	# Sharp - Equalized Histogram
	L_smooth_sharp = cv2.equalizeHist(L_smooth)
	R_smooth_sharp = cv2.equalizeHist(R_smooth)
	# Compute disparity (using method disparity_ncorr defined in disparity_ncorr.py)
	D_L = disparity_ncorr(L_smooth_sharp, R_smooth_sharp)
	D_R = disparity_ncorr(R_smooth_sharp, L_smooth_sharp)
	# Save output images
	# Note: They may need to be scaled/shifted before saving to show results properly
	D_L_equ = cv2.equalizeHist(D_L)
	cv2.imwrite('output/ps2-4-a-1.png', D_L_equ)
	D_R_equ = cv2.equalizeHist(D_R)
	cv2.imwrite('output/ps2-4-a-2.png', D_R_equ)
		
if __name__ == "__main__":
	# Ignore Python warnings
	warnings.filterwarnings("ignore")
	main()