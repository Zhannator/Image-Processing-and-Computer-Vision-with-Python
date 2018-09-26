import numpy as np
import cv2
import math

################################################################
# hough_lines_acc 
# Compute Hough accumulator array for finding lines.
#
# BW: Binary (black and white) image containing edge pixels
# RhoResolution (optional): Difference between successive rho values, in pixels
# Theta (optional): Vector of theta values to use, in degrees
#
# Matlab documentation for hough():
# http://www.mathworks.com/help/images/ref/hough.html
# 
# Note: Rows of H should correspond to values of rho, columns those of theta.
################################################################
def hough_lines_acc (BW, thetaResolution = 1, rhoResolution = 1): # (BW, varargin)	
	# Calculate maximum rho
	height, width = BW.shape
	rho_max = np.round(math.sqrt(math.pow(height, 2) + math.pow(width, 2)))
	
	# Maximum theta
	theta_max = 90
	
	# Initialize theta and rho array
	theta = np.arange((-1)*theta_max, theta_max, thetaResolution, 'double')
	rho = np.arange((-1)*rho_max, rho_max, rhoResolution, 'double')

	# Initialize H[rho, theta] = 0
	H = np.zeros((len(rho), len(theta)), np.uint64)
	
	# For each edge point in BW[row, column] check all possible lines that could go through it
	for row in range(height):
		for column in range(width):
			if (BW[row][column] > 0):
				for T in theta:
					# Calculate rho value for [row, column] with theta = T
					d = np.round(column * math.cos(math.radians(T)) + row * math.sin(math.radians(T)))
					d = d + rho_max - 1
					# Round rho to rhoResolution
					d_remainder = d % rhoResolution
					if (d_remainder != 0):
						d = d - d_remainder
						if (d_remainder >= (d_remainder/2)):
							d = d + rhoResolution
					# Account for rhoResolution != 1
					d = d / rhoResolution
					d = d.astype('uint64')
					
					# Round theta to rhoResolution
					T_index = (T + theta_max).astype('intc')
					T_index_remainder = T_index % thetaResolution
					if (T_index_remainder != 0):
						T_index = T_index - T_index_remainder
						if (T_index_remainder >= (T_index_remainder/2)):
							T_index = T_index + thetaResolution
					# Account for thetaResolution != 1
					T_index = T_index / thetaResolution
					T_index = T_index.astype('uint64')
					
					# Update H
					H[d][T_index] = H[d][T_index] + 1
	
	# Return [H, theta, rho]
	return H, theta, rho # {'H' : H, 'theta' : theta, 'rho' : rho}