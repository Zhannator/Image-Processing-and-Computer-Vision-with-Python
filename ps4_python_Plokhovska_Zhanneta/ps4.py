# Public libraries
import sys
import os
import numpy as np
import cv2
import math
import random

#############################################################
# Compute X and Y gradients for an image
#############################################################
def compute_gradient(img):
	# Sobel operator - approximation of derivative of Gaussian - (1/8) needed to get right gradient magnitude
	kernel_x = np.array([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]) / 8
	kernel_y = np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]) / 8

	rows, columns = img.shape
	
	# Pad image with zero's
	img_padded = np.pad(img, ((1, 1), (1, 1)), 'constant')
	
	# X and Y derivatives
	img_x = np.zeros((rows, columns), np.int64)
	img_y = np.zeros((rows, columns), np.int64)	
	for r in range(rows):
		for c in range(columns):
			img_part = img_padded[r : r + 3, c : c + 3]
			img_x[r, c] = (img_part * kernel_x).sum()
			img_y[r, c] = (img_part * kernel_y).sum()
	
	return (img_x + img_x.min()).astype(np.uint8), (img_y + img_y.min()).astype(np.uint8)

def main():
	
	# 1-A: Compute X and Y gradients
	print "\n-----------------------1-A-----------------------" 
	# Read images
	transA = cv2.imread(os.path.join('input', 'transA.jpg'), 0)  # grayscale
	simA = cv2.imread(os.path.join('input', 'simA.jpg'), 0)  # grayscale
	
	# Calculate x and y gradients
	#transA_x = cv2.Sobel(transA, cv2.CV_64F, 1, 0, ksize = 3)
	#transA_y = cv2.Sobel(transA, cv2.CV_64F, 0, 1, ksize = 3)
	#simA_x = cv2.Sobel(simA, cv2.CV_64F, 1, 0, ksize = 3)
	#simA_y = cv2.Sobel(simA, cv2.CV_64F, 0, 1, ksize = 3)
	transA_x, transA_y = compute_gradient(transA)
	simA_x, simA_y = compute_gradient(simA)
	
	# Horizontally concatenate x and y gradients into a single wide image
	transA_xy = cv2.hconcat([transA_x, transA_y])
	simA_xy = cv2.hconcat([simA_x, simA_y])
	
	# Save images
	cv2.imwrite('output/ps4-1-a-1.png', transA_xy)
	cv2.imwrite('output/ps4-1-a-2.png', simA_xy)

	# 1-B: Compute Harris value for the image
	print "\n-----------------------1-B-----------------------"
	
	
	
if __name__ == "__main__":
	main()