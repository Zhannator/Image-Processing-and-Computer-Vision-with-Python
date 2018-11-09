# Public libraries
import sys
import os
import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt

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
def find_corner_points(harris_img, threshold = 0.5, nHoodSize = [1, 1]):
	rows, columns = harris_img.shape
	
	# Optional to normalize
	harris_img = normalize(harris_img)
	
	# Default threshold
	max_row, max_column = np.unravel_index(np.argmax(harris_img), (rows, columns))
	min_row, min_column = np.unravel_index(np.argmin(harris_img), (rows, columns))
	threshold = threshold * (harris_img[max_row][max_column] + harris_img[min_row][min_column])
	#print "\nThreshold: {} Max: {} Min: {}\n".format(threshold, harris_img[max_row][max_column], harris_img[min_row][min_column])
	
	# Default nHoodSize
	height, width = harris_img.shape
	if nHoodSize  == [1, 1]:
		nHoodSize = [np.floor((height/100) * 2 + 1), np.floor((width/100) * 2 + 1)]
	nHoodSizeHalf = [(np.floor(nHoodSize[0]/2)).astype('int64'), (np.floor(nHoodSize[1]/2)).astype('int64')]
	#print "\nSize of neighborhood of supression: {}\n".format(nHoodSize)
	
	#print nHoodSize
	
	# Initialize peak indices numpeaks x 2 array
	Q = []
	
	numCorners = 0
	
	while(True):
	
		# Test on 10 points
		if numCorners == 10:
			break
	
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
			numCorners = numCorners + 1
		else:
			break
	#print "\nNumber of corners: {}\n".format(numCorners)
			
	return np.array(Q)

#################################################################################
# Translate feature points into SIFT keypoints using X and Y gradients 
#################################################################################
def compute_keypoints(points):
	# Compute an angle for every point
	angles = np.arctan2(points[:, 0], points[:, 1])
	keypoints = []
	# Create list of keypoints
	for i in range((points.shape)[0]):
		# Set the value of _octave to 0, since all points were located at the full scale version of the image
		point = cv2.KeyPoint(points[i][1], points[i][0], _size = 3, _angle = angles[i], _octave = 0)
		keypoints.append(point)
	
	return keypoints

#################################################################################
# Use matches and points from SIFT to draw lines connecting points in two images 
#################################################################################
def draw_matches(img1, points1, img2, points2, matches):
	rows, columns = img1.shape
	# Concatenate two images
	img_concat = cv2.hconcat([img1, img2])
	# Change image from gray to colored image
	img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)
	for m in matches:
		#print "\nIndex1: {} Index2: {}\n".format(m.queryIdx, m.trainIdx)
		#print "Point1: {}, {}".format(points1[index1].pt[0], points1[index1].pt[1])
		#print "Point2: {}, {}".format(points2[index2].pt[0], points2[index2].pt[1])
		# Index in points1 and points2
		index1 = m.queryIdx
		index2 = m.trainIdx
		# Index in img1 and img2
		p1_col = int(points1[index1].pt[0])
		p1_row = int(points1[index1].pt[1])
		p2_col = int(points2[index2].pt[0]) + columns
		p2_row = int(points2[index2].pt[1])
		# Draw key points
		cv2.circle(img_concat, (p1_col, p1_row), 3, (255, 0, 0), thickness = 1, lineType = 8) # Draw a circle using center coordinates and radius
		cv2.circle(img_concat, (p2_col, p2_row), 3, (255, 0, 0), thickness = 1, lineType = 8) # Draw a circle using center coordinates and radius
		# Draw line (image, (column, row), (column, row), color, thickness)
		cv2.line(img_concat, (p1_col, p1_row), (p2_col, p2_row), (255, 0, 0), 1)
		
	return img_concat

def calculate_residual(point1, point2):
	# You can do the comparison by checking the residual
	# between the predicted location of each test point using your equation and the actual location
	# given by the 2D input data. The residual is just the distance (square root of the sum of squared
	# differences in u and v).
	return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

#################################################################################
# Ransac_translation returns matches for translation images with best consensus
# s - minimal set (minimal number to fit the model)
# t - distance threshold (needs noise model)
# N - number of samples (Choose N so that, with probability p, at least one
#     random sample set is free from outliers (e.g. p = 0.99))
#################################################################################
def ransac_translation(points1, points2, matches):	
	# Calculate N (number of samples)
	s = 2 # minimal set (minimal number to fit the model)
	p = 0.99 # probability of success - designer's choice
	e = 0.05 # proportion of outliers, inliers = (1 - e) - designer's choice
	N = int(math.log(1 - p, 2) / math.log(1 - math.pow(1 - e, s)))
	#print "\nN: {}\n".format(N)
	# Calculate t (threshold for residual)
	t = 200

	matches_len = len(matches)
	offsets = []
	# Calculate offset for every match
	for i in range(matches_len):
		m = matches[i]
		# Offsets
		col_offset = int(points1[m.queryIdx].pt[0] - points2[m.trainIdx].pt[0])
		row_offset = int(points1[m.queryIdx].pt[1] - points2[m.trainIdx].pt[1])
		offsets.append([col_offset, row_offset])
	offsets = np.array(offsets)

	# Randomly select N of the putative matches - calculate offset (a translation in X and Y ) between the two images
	# Find out how many other putative matches agree with this offset (account for noise)
	random_matches = []
	if N > matches_len:
		random_matches = random.sample(range(0, matches_len), matches_len)
	else:
		random_matches = random.sample(range(0, matches_len), N)
	best_consensus = [] # initially no matches
	best_consensus_count = 0 # initially no matches
	best_offset = []
	for i in random_matches:
		# Offsets
		offset = offsets[i]
		
		matches_new = []
		matches_new_count = 0
		for i_new in range(matches_len):
			# Offsets
			offset_residual = calculate_residual(offset, offsets[i_new])
			# Check that the difference in offsets (residual) is not too large
			if (offset_residual < t):
				matches_new.append(matches[i_new])
				matches_new_count = matches_new_count + 1
				
		if best_consensus_count < matches_new_count:
			best_consensus = matches_new
			best_consensus_count = matches_new_count
			best_offset = offset
	
	#print "\nBest offset: {}\n".format(best_offset)
	return best_consensus
'''
#################################################################################
# Ransac_similarity returns matches for translation images with best consensus
# s - minimal set (minimal number to fit the model)
# t - distance threshold (needs noise model)
# N - number of samples (Choose N so that, with probability p, at least one
#     random sample set is free from outliers (e.g. p = 0.99))
#################################################################################
def ransac_similarity(points1, points2, matches):	
	# Calculate N (number of samples)
	s = 4 # minimal set (minimal number to fit the model)
	p = 0.99 # probability of success - designer's choice
	e = 0.05 # proportion of outliers, inliers = (1 - e) - designer's choice
	N = int(math.log(1 - p, 2) / math.log(1 - math.pow(1 - e, s)))
	#print "\nN: {}\n".format(N)
	# Calculate t (threshold for residual)
	t = 200

	matches_len = len(matches)

	# Randomly select N of the putative matches - calculate offset (a translation in X and Y ) between the two images
	# Find out how many other putative matches agree with this offset (account for noise)
	best_consensus = [] # initially no matches
	best_consensus_count = 0 # initially no matches
	best_offset = []
	for i in range(N):
		random_matches = random.sample(range(0, matches_len), matches_len - (matches_len % 2))
		
		###############
		TODOOOOOOOOOOO
		###############
		
		matches_new = []
		matches_new_count = 0
		for i_new in range(matches_len):
			# Offsets
			offset_residual = calculate_residual(offset, offsets[i_new])
			# Check that the difference in offsets (residual) is not too large
			if (offset_residual < t):
				matches_new.append(matches[i_new])
				matches_new_count = matches_new_count + 1
				
		if best_consensus_count < matches_new_count:
			best_consensus = matches_new
			best_consensus_count = matches_new_count
			best_offset = offset
	
	#print "\nBest offset: {}\n".format(best_offset)
	return best_consensus
'''
#################################################################################
# Solves for similarity transform using 2 pair 
#################################################################################
def solve_similarity_transform(point1_1, point1_2, point2_1, point2_2):
	# Create 4 by 5 matrix to solve using Gaussian elimination
	A = np.array([[point1_1[0], -point1_1[1], 1, 0], 
				  [point1_1[1], point1_1[0], 0, 1], 
				  [point2_1[0], -point2_1[1], 1, 0], 
				  [point2_1[1], point2_1[0], 0, 1]])
	b = np.array([point1_2[0], point1_2[1], point2_2[0], point2_2[1]])
	solution = numpy.linalg.solve(A, b) # [a, b, c, d]
	similarity_transform = np.array([[solution[0], -solution[1], solution[2]], [solution[1], solution[0], solution[3]]])
	return similarity_transform
	
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
	corner_points = find_corner_points(test_harris_image, 0.5, [5, 5])
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	test = cv2.cvtColor(test,cv2.COLOR_GRAY2RGB)
	for i in range(len(corner_points)):
		cv2.circle(test, (corner_points[i][1], corner_points[i][0]), 3, (0, 0, 255), thickness=1, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/test_corners.png', test)
	'''
	# transA
	transA_corner_points = find_corner_points(transA_harris_image, 0.35)
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	transA_draw_corner_points = cv2.cvtColor(transA, cv2.COLOR_GRAY2RGB)
	for i in range(len(transA_corner_points)):
		cv2.circle(transA_draw_corner_points, (transA_corner_points[i][1], transA_corner_points[i][0]), 3, (0, 0, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/ps4-1-c-1.png', transA_draw_corner_points)
	# transB
	transB_corner_points = find_corner_points(transB_harris_image, 0.30)
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	transB_draw_corner_points = cv2.cvtColor(transB, cv2.COLOR_GRAY2RGB)
	for i in range(len(transB_corner_points)):
		cv2.circle(transB_draw_corner_points, (transB_corner_points[i][1], transB_corner_points[i][0]), 3, (0, 0, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/ps4-1-c-2.png', transB_draw_corner_points)
	# simA
	simA_corner_points = find_corner_points(simA_harris_image, 0.25)
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	simA_draw_corner_points = cv2.cvtColor(simA,cv2.COLOR_GRAY2RGB)
	for i in range(len(simA_corner_points)):
		cv2.circle(simA_draw_corner_points, (simA_corner_points[i][1], simA_corner_points[i][0]), 3, (0, 0, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/ps4-1-c-3.png', simA_draw_corner_points)
	# simB
	simB_corner_points = find_corner_points(simB_harris_image, 0.20)
	# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
	simB_draw_corner_points = cv2.cvtColor(simB,cv2.COLOR_GRAY2RGB)
	for i in range(len(simB_corner_points)):
		cv2.circle(simB_draw_corner_points, (simB_corner_points[i][1], simB_corner_points[i][0]), 3, (0, 0, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
	# Plot/show accumulator array H, save as output/ps1-2-a-1.png
	cv2.imwrite('output/ps4-1-c-4.png', simB_draw_corner_points)
	
	# 2-A: SIFT - Interest points an angles
	print "\n-----------------------2-A-----------------------"
	# Calculate angles for every keypoint pair in every image
	transA_keypoints = compute_keypoints(transA_corner_points)
	transB_keypoints = compute_keypoints(transB_corner_points)
	simA_keypoints = compute_keypoints(simA_corner_points)
	simB_keypoints = compute_keypoints(simB_corner_points)
	# Draw keypoints on image
	transA_draw_keypoints = cv2.drawKeypoints(transA, transA_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	transB_draw_keypoints = cv2.drawKeypoints(transB, transB_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	plt.subplot(121),plt.imshow(transA_draw_keypoints)
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(transB_draw_keypoints)
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

	transAB_with_keypoints = cv2.hconcat([transA_draw_keypoints, transB_draw_keypoints])
	cv2.imwrite('output/ps4-2-a-1.png', transAB_with_keypoints)
	simA_draw_keypoints = cv2.drawKeypoints(simA, simA_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	simB_draw_keypoints = cv2.drawKeypoints(simB, simB_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	simAB_with_keypoints = cv2.hconcat([simA_draw_keypoints, simB_draw_keypoints])
	cv2.imwrite('output/ps4-2-a-2.png', simAB_with_keypoints)
	
	# 2-B: SIFT - Interest points an angles
	print "\n-----------------------2-B-----------------------"
	# Create an instance of the class cv2.SIFT
	sift = cv2.xfeatures2d.SIFT_create()
	# Create BFMatcher instance
	bf = cv2.BFMatcher()
	# Extracting the SIFT descriptors then requires to call the SIFT.compute() function:
	transA_points, transA_descriptors = sift.compute(transA ,transA_keypoints)
	transB_points, transB_descriptors = sift.compute(transB ,transB_keypoints)
	simA_points, simA_descriptors = sift.compute(simA ,simA_keypoints)
	simB_points, simB_descriptors = sift.compute(simB , simB_keypoints)
	# Match descriptors
	trans_matches = bf.match(transA_descriptors, transB_descriptors)
	sim_matches = bf.match(simA_descriptors, simB_descriptors)
	# Sort them in order of distance
	trans_matches = sorted(trans_matches, key = lambda x:x.distance)
	sim_matches = sorted(sim_matches, key = lambda x:x.distance)
	# Draw lines connecting points in two images
	#trans_putative_pair_image = cv2.drawMatches(transA, transA_points, transB, transB_points, trans_matches, None, flags=2)
	#sim_putative_pair_image = cv2.drawMatches(simA, simA_points, simB, simB_points, sim_matches, None, flags=2)
	trans_putative_pair_image = draw_matches(transA, transA_points, transB, transB_points, trans_matches)
	sim_putative_pair_image = draw_matches(simA, simA_points, simB, simB_points, sim_matches)
	# Save images
	cv2.imwrite('output/ps4-2-b-1.png', trans_putative_pair_image)
	cv2.imwrite('output/ps4-2-b-2.png', sim_putative_pair_image)
	
	# 3-A: RANSAC - Translational case
	print "\n-----------------------3-A-----------------------"
	trans_best_matches = ransac_translation(transA_points, transB_points, trans_matches)
	trans_putative_pair_image = draw_matches(transA, transA_points, transB, transB_points, trans_best_matches)
	cv2.imwrite('output/ps4-3-a-1.png', trans_putative_pair_image)
	'''
	# 3-B: RANSAC - Similarity transform - similarity transform allows translation, rotation and scaling
	print "\n-----------------------3-B-----------------------"
	sim_best_matches = ransac_similarity(simA_points, simB_points, sim_matches)
	sim_putative_pair_image = draw_matches(simA, simA_points, simB, simB_points, sim_best_matches)
	cv2.imwrite('output/ps4-3-a-1.png', sim_putative_pair_image)
	'''
	
if __name__ == "__main__":
	main()