# Public libraries
import sys
import os
import numpy as np
import cv2
import math
import random

#################################################################################
# Normalize image
#################################################################################
def normalize(img, distribute = True):
	img = img - img.min() # Get rid of negative values
	if distribute == True:
		img = (img * (1/img.max()) * 255) # distribute values between 0 and 255
	return img.astype('uint8')

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
# Compute Harris image for an X and Y gradients of an image
#################################################################################
def compute_harris_image(Ix, Iy):
	rows, columns = Ix.shape
	Ix_padded = np.pad(Ix, ((1, 1), (1, 1)), 'constant')
	Iy_padded = np.pad(Iy, ((1, 1), (1, 1)), 'constant')
	Ix_Ix = Ix_padded * Ix_padded
	Iy_Iy = Iy_padded * Iy_padded
	Ix_Iy = Ix_padded * Iy_padded
	w = np.ones((3, 3)) # Designer's choice: can be ones or gaussian, size of w
	alfa = 0.04 # Designers choice withing range(0.04, 0.06)
	harris_image = np.zeros((rows, columns))
	for r in range(rows):
		for c in range(columns):
			Ix_Ix_temp = Ix_Ix[r : r + 3, c : c + 3]
			Iy_Iy_temp = Iy_Iy[r : r + 3, c : c + 3]
			Ix_Iy_temp = Ix_Iy[r : r + 3, c : c + 3]
			M = np.array([[(w*Ix_Ix_temp).sum(), (w*Ix_Iy_temp).sum()], [(w*Ix_Iy_temp).sum(), (w*Iy_Iy_temp).sum()]])
			R = np.linalg.det(M) - alfa * math.pow(np.matrix.trace(M), 2)
			harris_image[r, c] = R
	return harris_image

#################################################################################
# Find corner points in Harris image using threshold and non-maximal suppression 
#################################################################################
def find_corner_points(harris_img, numcorners = 300, threshold = 0.5, nHoodSize = [1, 1]):
	rows, columns = harris_img.shape
	
	# Optional to normalize
	harris_img = normalize(harris_img)
	
	# Default threshold
	max_row, max_column = np.unravel_index(np.argmax(harris_img), (rows, columns))
	min_row, min_column = np.unravel_index(np.argmin(harris_img), (rows, columns))
	threshold = threshold * (harris_img[max_row][max_column] + harris_img[min_row][min_column])
	print "\nThreshold: {} Max: {} Min: {}\n".format(threshold, harris_img[max_row][max_column], harris_img[min_row][min_column])
	
	# Default nHoodSize
	height, width = harris_img.shape
	if nHoodSize  == [1, 1]:
		nHoodSize = [np.floor((height/100) * 2 + 1), np.floor((width/100) * 2 + 1)]
	nHoodSizeHalf = [(np.floor(nHoodSize[0]/2)).astype('int64'), (np.floor(nHoodSize[1]/2)).astype('int64')]
	#print "\nSize of neighborhood of supression: {}\n".format(nHoodSize)
	
	#print nHoodSize
	
	# Initialize peak indices numpeaks x 2 array
	Q = []

	for i in range(numcorners):
		# Find maximum value in Hough
		max_row, max_column = np.unravel_index(np.argmax(harris_img), (rows, columns)) # Returns location of maximum value in harris_img
		#print str(max_row) + " " + str(max_column)
		#print harris_img[max_row][max_column]
		
		# Make sure max value is >= threshold
		if harris_img[max_row][max_column] >= threshold:
			# Suppress area around max in the neighborhood
			# Two-element vector of positive odd integers: [M N]. 'NHoodSize' specifies the size of the suppression neighborhood.
			for R in range((max_row - nHoodSizeHalf[0]).astype('int64'), max_row + nHoodSizeHalf[0] + 1, 1):
				if ((R >= 0) & (R < height)):
					for C in range((max_column - nHoodSizeHalf[1]).astype('int64'), max_column + nHoodSizeHalf[1] + 1, 1):
						if ((C >= 0) & (C < width)):
							harris_img[R][C] = threshold - 1
			
			# Add peak to Q		
			Q.append([max_row, max_column])
		else:
			break
	#print "\nNumber of corners: {}\n".format(i - 1)
			
	return Q
	
def main():
	# 1-A: Compute X and Y gradients
	print "\n-----------------------1-A-----------------------" 
	# Read images
	transA = cv2.imread(os.path.join('input', 'transA.jpg'), 0)  # grayscale
	simA = cv2.imread(os.path.join('input', 'simA.jpg'), 0)  # grayscale	
	# Calculate x and y gradients
	transA_x, transA_y = compute_gradient(transA)
	simA_x, simA_y = compute_gradient(simA)
	# Horizontally concatenate x and y gradients into a single wide image
	transA_xy = cv2.hconcat([normalize(transA_x, False), normalize(transA_y, False)])
	simA_xy = cv2.hconcat([normalize(simA_x, False), normalize(simA_y, False)])
	# Save images
	cv2.imwrite('output/ps4-1-a-1.png', transA_xy)
	cv2.imwrite('output/ps4-1-a-2.png', simA_xy)

	# 1-B: Compute Harris value for the image
	print "\n-----------------------1-B-----------------------"
	'''
	# Test on check
	test = cv2.imread(os.path.join('input', 'check_rot.bmp'), 0)  # grayscale
	test_x, test_y = compute_gradient(test)
	cv2.imwrite('output/test_gradient.png', cv2.hconcat([normalize(test_x, False), normalize(test_y, False)]))
	test_harris_image = compute_harris_image(test_x, test_y)
	cv2.imwrite('output/test_harris.png', normalize(test_harris_image))
	'''
	# Read images
	transB = cv2.imread(os.path.join('input', 'transB.jpg'), 0)  # grayscale
	simB = cv2.imread(os.path.join('input', 'simB.jpg'), 0)  # grayscale	
	# Calculate gradients fro transB and simB, gradients for transA and simA are calculated above
	transB_x, transB_y = compute_gradient(transB)
	simB_x, simB_y = compute_gradient(simB)
	# Calculate Harris value (R) for every pixel
	transA_harris_image = compute_harris_image(transA_x, transA_y)
	transB_harris_image = compute_harris_image(transB_x, transB_y)
	simA_harris_image = compute_harris_image(simA_x, simA_y)
	simB_harris_image = compute_harris_image(simB_x, simB_y)
	# Save Harris image
	cv2.imwrite('output/ps4-1-b-1.png', normalize(transA_harris_image))
	cv2.imwrite('output/ps4-1-b-2.png', normalize(transB_harris_image))
	cv2.imwrite('output/ps4-1-b-3.png', normalize(simA_harris_image))
	cv2.imwrite('output/ps4-1-b-4.png', normalize(simB_harris_image))
	
	# 1-C: Find corner points - thresholding and non-maximal suppression
	print "\n-----------------------1-C-----------------------"
	'''
	# Test on check
	corner_points = find_corner_points(test_harris_image, 25, 0.5, [5, 5])
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	test = cv2.cvtColor(test,cv2.COLOR_GRAY2RGB)
	for i in range(len(corner_points)):
		cv2.circle(test, (corner_points[i][1], corner_points[i][0]), 3, (0, 0, 255), thickness=1, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/test_corners.png', test)
	'''
	# transA
	corner_points = find_corner_points(transA_harris_image, 300, 0.05)
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	transA = cv2.cvtColor(transA, cv2.COLOR_GRAY2RGB)
	for i in range(len(corner_points)):
		cv2.circle(transA, (corner_points[i][1], corner_points[i][0]), 3, (0, 0, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/ps4-1-c-1.png', transA)
	# transB
	corner_points = find_corner_points(transB_harris_image, 300, 0.05)
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	transB = cv2.cvtColor(transB, cv2.COLOR_GRAY2RGB)
	for i in range(len(corner_points)):
		cv2.circle(transB, (corner_points[i][1], corner_points[i][0]), 3, (0, 0, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/ps4-1-c-2.png', transB)
	# simA
	corner_points = find_corner_points(simA_harris_image, 300, 0.05)
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	simA = cv2.cvtColor(simA,cv2.COLOR_GRAY2RGB)
	for i in range(len(corner_points)):
		cv2.circle(simA, (corner_points[i][1], corner_points[i][0]), 3, (0, 0, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/ps4-1-c-3.png', simA)
	# simB
	corner_points = find_corner_points(simB_harris_image, 300, 0.05)
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	simB = cv2.cvtColor(simB,cv2.COLOR_GRAY2RGB)
	for i in range(len(corner_points)):
		cv2.circle(simB, (corner_points[i][1], corner_points[i][0]), 3, (0, 0, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/ps4-1-c-4.png', simB)
	
	# 2-A: Interest points an angles
	print "\n-----------------------1-C-----------------------"
	
	
if __name__ == "__main__":
	main()