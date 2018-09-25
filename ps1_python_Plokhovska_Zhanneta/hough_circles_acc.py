import numpy as np
import cv2
import math

################################################################
# hough_circles_acc
# Compute Hough accumulator array for finding circles.
#	
# BW: Binary (black and white) image containing edge pixels
# radius: Radius of circles to look for, in pixels
################################################################
def hough_circles_acc (BW, radius):
	height, width = BW.shape
	thetaMax = 360
	
	H = np.zeros((height, width), np.uint64)
	# For each edge point in BW[row, column] check all possible lines that could go through it
	for row in range(height):
		for column in range(width):
			if (BW[row][column] > 1):
				for theta in range(thetaMax):
					# Calculate coordinates for point on the circle at theta degrees
					R = (np.round(row + radius * np.sin(theta))).astype('int64')
					C = (np.round(column + radius * np.cos(theta))).astype('int64')
					# Update H
					if ((R >= 0) & (R < height) & (C >= 0) & (C < width)):
						H[R][C] = H[R][C] + 1
	
	return H