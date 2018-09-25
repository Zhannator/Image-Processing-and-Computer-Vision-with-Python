import numpy as np
import cv2
import math

# Private libraries
from hough_peaks import hough_peaks
from hough_circles_acc import hough_circles_acc

################################################################
# find_circles
# Find circles in given radius range using Hough transform.
#
# BW: Binary (black and white) image containing edge pixels
# radius_range: Range of circle radii [min, max] to look for, in pixels
################################################################
def find_circles(BW, radius_range):
	
	centers = []
	radii = []
	
	for radius in range(radius_range[0], radius_range[1]):
		# Find hough transform
		H = hough_circles_acc(BW, radius)
		# Find circle centers
		num_of_circles = 5
		current_centers = hough_peaks(H, num_of_circles, [30, 30], 120)
		
		for i in range(len(current_centers)):
			centers.append(current_centers[i])
			radii.append(radius)
   
	return [centers, radii]