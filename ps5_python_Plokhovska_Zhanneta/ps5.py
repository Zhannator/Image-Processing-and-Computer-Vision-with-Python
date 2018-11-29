# Public libraries
import sys
import os
import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt
import itertools

#################################################################################
# Compute X and Y gradients for an image
#################################################################################
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

	return img_x, img_y

#################################################################################
# Compute LK optic flow estimation using X/Y gradients of image 1 and image difference
#################################################################################
def lk_optic_flow(Ix, Iy, It):
	rows, columns = Ix.shape
	Ix_padded = np.pad(Ix, ((2, 2), (2, 2)), 'constant')
	Iy_padded = np.pad(Iy, ((2, 2), (2, 2)), 'constant')
	It_padded = np.pad(It, ((2, 2), (2, 2)), 'constant')
	Ix_Ix = Ix_padded * Ix_padded
	Iy_Iy = Iy_padded * Iy_padded
	Ix_Iy = Ix_padded * Iy_padded
	Ix_It = Ix_padded * It_padded
	Iy_It = Iy_padded * It_padded
	w = np.ones((5, 5)) # Designer's choice: can be ones or gaussian, size of w
	alfa = 0.04 # Designers choice withing range(0.04, 0.06)
	u_image = np.zeros((rows, columns))
	v_image = np.zeros((rows, columns))
	for r in range(rows):
		for c in range(columns):
			S_Ix_Ix = (Ix_Ix[r : r + 5, c : c + 5]).sum()
			S_Iy_Iy = (Iy_Iy[r : r + 5, c : c + 5]).sum()
			S_Ix_Iy = (Ix_Iy[r : r + 5, c : c + 5]).sum()
			S_Ix_It = (Ix_It[r : r + 5, c : c + 5]).sum()
			S_Iy_It = (Iy_It[r : r + 5, c : c + 5]).sum()
			A = np.array([[S_Ix_Ix, S_Ix_Iy],  
						  [S_Ix_Iy, S_Iy_Iy]])
			b = np.array([-S_Ix_It, -S_Iy_It])
			try:
				solution = np.linalg.solve(A, b) # [u, v]
			except:
				solution = [0, 0]
			u_image[r, c] = solution[0]
			v_image[r, c] = solution[1]
	
	return u_image, v_image

#################################################################################
# Normalize image
#################################################################################
def normalize(img, distribute = True):
	img = img - img.min() # Get rid of negative values
	if distribute == True:
		img = (img * (1/img.max()) * 255) # distribute values between 0 and 255
	return img.astype('uint8')

#################################################################################
# Gaussian Reduce Function (downsamples image 2x)
#################################################################################
def reduce(img):
	rows, columns = img.shape
	# img must always have odd number of rows and columns
	img = cv2.GaussianBlur(img, (5, 5), 0)
	img_reduced = img[::2, 1::2] # select all odd rows
	rows_reduced, columns_reduced = img_reduced.shape
	#print "Reduced Shape: {}".format(img_reduced.shape)
	
	return img_reduced

#################################################################################
# Laplacian Expand Function (upsamples image 2x)
#################################################################################
def expand(img, shape):
	rows, columns = img.shape
	img_expanded = np.zeros((shape[0], shape[1])) # img must always have odd number of rows and columns
	rows_expanded, columns_expanded = img_expanded.shape
	for r in range(rows):
		for c in range(columns):
			img_expanded[r * 2, c * 2] = img[r, c]
	img_expanded = cv2.GaussianBlur(img_expanded, (5, 5), 0)
	#print "Expanded Shape: {}".format(img_expanded.shape)
	
	return img_expanded

#################################################################################
# 4-level Laplacian Pyramid 
#################################################################################
def laplacian_pyramid(img):
	# Reduce images
	img_reduced1 = reduce(img)
	img_reduced2 = reduce(img_reduced1)
	img_reduced3 = reduce(img_reduced2)
	img_reduced4 = reduce(img_reduced3)
	# Expand images
	img_expanded1 = expand(img_reduced4, img_reduced3.shape)
	img_expanded2 = expand(img_expanded1, img_reduced2.shape)
	img_expanded3 = expand(img_expanded2, img_reduced1.shape)
	# Calculate laplacian pyramid
	gauss = img_reduced4
	laplac1 = img_expanded1 - img_reduced3
	laplac2 = img_expanded2 - img_reduced2
	laplac3 = img_expanded3 - img_reduced1
	
	return gauss, laplac1, laplac2, laplac3
	
#################################################################################
# Warp Function
#################################################################################
def warp(i2, vx, vy):
	# warp i2 according to flow field in vx vy 
	# this is a "backwards" warp: 
	# if vx,vy are correct then warpI2 == i1 
	M, N = i2.shape 
	x, y = np.meshgrid(range(0, N), range(0, M))
	# use Python interpolation routine
	#warpI3 = np.zeros((M, N))
	warpI3 = cv2.remap(i2, x + vx, y + vy, cv2.INTER_NEAREST) # warpI3 = interp2(x,y,i2,x+vx,y+vy,'*nearest')
	# use Python interpolation routine 
	warpI2 = cv2.remap(i2, x + vx, y + vy, cv2.INTER_LINEAR) # warpI2 = interp2(x,y,i2,x+vx,y+vy,'*linear')
	I = np.nonzero(np.isnan(warpI2))
	warpI2[I] = warpI3[I]
	
	return warpI2
	
def main():
	if False:
		# 1-A: LK optic flow estimation
		print "\n-----------------------1-A-----------------------" 
		# Read images
		shift0 = cv2.imread(os.path.join('input/TestSeq', 'Shift0.png'), 0)
		shiftr2 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR2.png'), 0)
		shiftr5u5 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR5U5.png'), 0)
		shiftr10 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR10.png'), 0)
		shiftr20 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR20.png'), 0)
		shiftr40 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR40.png'), 0)
		# Smooth images
		shift0 = cv2.GaussianBlur(shift0, (5, 5), 0)
		shiftr2 = cv2.GaussianBlur(shiftr2, (5, 5), 0)
		shiftr5u5 = cv2.GaussianBlur(shiftr5u5, (5, 5), 0)
		shiftr10 = cv2.GaussianBlur(shiftr10, (5, 5), 0)
		shiftr20 = cv2.GaussianBlur(shiftr20, (5, 5), 0)
		shiftr40 = cv2.GaussianBlur(shiftr40, (5, 5), 0)
		# Get image gradients
		shift0_Ix, shift0_Iy = compute_gradient(shift0)
		# Get image difference
		shift0_shiftr2_It = shiftr2 - shift0
		shift0_shiftr5u5_It = shiftr5u5 - shift0
		# Calculate displacement images using LK optic flow estimation
		shift0_shiftr2_u, shift0_shiftr2_v = lk_optic_flow(shift0_Ix, shift0_Iy, shift0_shiftr2_It)
		shift0_shiftr5u5_u, shift0_shiftr5u5_v = lk_optic_flow(shift0_Ix, shift0_Iy, shift0_shiftr5u5_It)
		# Save images
		shift0_shiftr2 = cv2.hconcat([normalize(shift0_shiftr2_u, False), normalize(shift0_shiftr2_v, False)])
		plt.imshow(shift0_shiftr2);
		plt.colorbar()
		plt.savefig('output/ps5-1-a-1.png')
		shift0_shiftr5u5 = cv2.hconcat([normalize(shift0_shiftr5u5_u, False), normalize(shift0_shiftr5u5_v, False)])
		plt.imshow(shift0_shiftr5u5);
		plt.savefig('output/ps5-1-a-2.png')
		
		# 1-B: LK optic flow estimation
		print "\n-----------------------1-B-----------------------" 
		# Get image difference
		shift0_shiftr10_It = shiftr10 - shift0
		shift0_shiftr20_It = shiftr20 - shift0
		shift0_shiftr40_It = shiftr40 - shift0
		# Calculate displacement images using LK optic flow estimation
		shift0_shiftr10_u, shift0_shiftr10_v = lk_optic_flow(shift0_Ix, shift0_Iy, shift0_shiftr10_It)
		shift0_shiftr20_u, shift0_shiftr20_v = lk_optic_flow(shift0_Ix, shift0_Iy, shift0_shiftr20_It)
		shift0_shiftr40_u, shift0_shiftr40_v = lk_optic_flow(shift0_Ix, shift0_Iy, shift0_shiftr40_It)
		# Save images
		shift0_shiftr10 = cv2.hconcat([normalize(shift0_shiftr10_u, False), normalize(shift0_shiftr10_v, False)])
		plt.imshow(shift0_shiftr10);
		plt.colorbar()
		plt.savefig('output/ps5-1-b-1.png')
		shift0_shiftr20 = cv2.hconcat([normalize(shift0_shiftr20_u, False), normalize(shift0_shiftr20_v, False)])
		plt.imshow(shift0_shiftr20);
		plt.savefig('output/ps5-1-b-2.png')
		shift0_shiftr40 = cv2.hconcat([normalize(shift0_shiftr40_u, False), normalize(shift0_shiftr40_v, False)])
		plt.imshow(shift0_shiftr40);
		plt.savefig('output/ps5-1-b-3.png')
	
	# 2-A: Gaussian and Laplacian Pyramids - Reduce
	print "\n-----------------------2-A-----------------------" 
	# Read images
	yos1 = cv2.imread(os.path.join('input/DataSeq1', 'yos_img_01.jpg'), 0)
	yos2 = cv2.imread(os.path.join('input/DataSeq1', 'yos_img_02.jpg'), 0)
	yos3 = cv2.imread(os.path.join('input/DataSeq1', 'yos_img_03.jpg'), 0)
	# Reduce images
	yos1_reduced1 = reduce(yos1)
	yos1_reduced2 = reduce(yos1_reduced1)
	yos1_reduced3 = reduce(yos1_reduced2)
	yos1_reduced4 = reduce(yos1_reduced3)
	# Save images
	fig = plt.figure()
	plt.subplot(711)
	plt.imshow(yos1_reduced1);
	plt.title('1/2')
	plt.subplot(713)
	plt.imshow(yos1_reduced2);
	plt.title('1/4')
	plt.subplot(715)
	plt.imshow(yos1_reduced3);
	plt.title('1/8')
	plt.subplot(717)
	plt.imshow(yos1_reduced4);
	plt.title('1/16')
	plt.savefig('output/ps5-2-a-1.png')
	
	# 2-B: Gaussian and Laplacian Pyramids - Expand
	print "\n-----------------------2-B-----------------------"
	# Expand images
	gauss, laplac1, laplac2, laplac3 = laplacian_pyramid(yos1)
	# Save images
	fig = plt.figure()
	plt.subplot(711)
	plt.imshow(gauss);
	plt.title('Gaussian')
	plt.subplot(713)
	plt.imshow(laplac1);
	plt.title('Laplacian 1')
	plt.subplot(715)
	plt.imshow(laplac2);
	plt.title('Laplacian 2')
	plt.subplot(717)
	plt.imshow(laplac3);
	plt.title('Laplacian 3')
	plt.savefig('output/ps5-2-b-1.png')
	
	# 3-A: Warping by flow
	print "\n-----------------------3-A-----------------------"
	# x and y displacements for Yos1 and Yos2
	yos1_reduced = yos1
	yos2_reduced = yos2
	yos3_reduced = yos3
	for i in range(1):
		yos1_reduced = reduce(yos1_reduced)
		yos2_reduced = reduce(yos2_reduced)
		yos3_reduced = reduce(yos3_reduced)
	yos1_Ix, yos1_Iy = compute_gradient(yos1_reduced)
	yos1_yos2_It = yos2_reduced - yos1_reduced
	yos1_yos2_u, yos1_yos2_v = lk_optic_flow(yos1_Ix, yos1_Iy, yos1_yos2_It)
	yos1_yos2 = cv2.hconcat([normalize(yos1_yos2_u, False), normalize(yos1_yos2_v, False)])
	# x and y displacements for Yos2 and Yos3
	yos2_Ix, yos2_Iy = compute_gradient(yos2_reduced)
	yos2_yos3_It = yos3_reduced - yos2_reduced
	yos2_yos3_u, yos2_yos3_v = lk_optic_flow(yos2_Ix, yos2_Iy, yos2_yos3_It)
	yos2_yos3 = cv2.hconcat([normalize(yos2_yos3_u, False), normalize(yos2_yos3_v, False)])
	# Save images
	fig = plt.figure()
	plt.subplot(411)
	plt.imshow(yos1_yos2);
	plt.title('DataSeq1: 1 to 2')
	plt.colorbar()
	plt.subplot(413)
	plt.imshow(yos2_yos3);
	plt.title('DataSeq1: 2 to 3')
	plt.colorbar()
	plt.savefig('output/ps5-3-a-1.png')
	# Difference image between warped image 2 and original image 1
	#yos2_warp = warp(yos2_reduced, yos1_yos2_u, yos1_yos2_v)
	yos2_warp = cv2.remap(yos2_reduced, (yos1_yos2_u).astype(CV_32FC1), (yos1_yos2_v).astype(CV_32FC1), cv2.INTER_NEAREST) # warpI3 = interp2(x,y,i2,x+vx,y+vy,'*nearest')
	yos2_warp_yos_1_difference = yos2_warp - yos1_reduced
	cv2.imwrite('output/ps5-3-a-2.png', yos2_warp_yos_1_difference)
	
if __name__ == "__main__":
	main()