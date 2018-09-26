import numpy as np
import cv2
import math

################################################################
# hough_peaks
# Find peaks in a Hough accumulator array.
#
# Threshold (optional): Threshold at which values of H are considered to be peaks
# NHoodSize (optional): Size of the suppression neighborhood, [M N]
#
# Matlab documentation for houghpeaks():
# http://www.mathworks.com/help/images/ref/houghpeaks.html
################################################################
def hough_peaks(H, numpeaks = 6, nHoodSize = [1, 1], threshold = 0, theta = np.arange(-90, 90, 1, 'double')): # (H, varargin)
	
	# Default threshold
	max_H_row, max_H_column = np.unravel_index(np.argmax(H), H.shape)
	if threshold == 0:
		threshold = 0.5 * H[max_H_row][max_H_column]
	print "MAX: " + str(H[max_H_row][max_H_column])
	
	#print threshold
	
	# Default nHoodSize
	height, width = H.shape
	if nHoodSize  == [1, 1]:
		nHoodSize = [np.floor((height/100) * 2 + 1), np.floor((width/100) * 2 + 1)]
	nHoodSizeHalf = [(np.floor(nHoodSize[0]/2)).astype('int64'), (np.floor(nHoodSize[1]/2)).astype('int64')]
	
	#print nHoodSize
	
	# Initialize peak indices numpeaks x 2 array
	Q = []

	for i in range(numpeaks):
		# Find maximum value in Hough
		max_H_row, max_H_column = np.unravel_index(np.argmax(H), H.shape) # Returns location of maximum value in H
		#print str(max_H_row) + " " + str(max_H_column)
		#print H[max_H_row][max_H_column]
		
		# Make sure max value is >= threshold
		if H[max_H_row][max_H_column] >= threshold:
			# Suppress area around max_H in the neighborhood
			# Two-element vector of positive odd integers: [M N]. 'NHoodSize' specifies the size of the suppression neighborhood.
			for R in range((max_H_row - nHoodSizeHalf[0]).astype('int64'), max_H_row + nHoodSizeHalf[0] + 1, 1):
				if ((R >= 0) & (R < height)):
					for C in range((max_H_column - nHoodSizeHalf[1]).astype('int64'), max_H_column + nHoodSizeHalf[1] + 1, 1):
						if ((C >= 0) & (C < width)):
							H[R][C] = 0
			
			# Add peak to Q		
			Q.append([max_H_row, max_H_column])
		else:
			break
			
	return Q