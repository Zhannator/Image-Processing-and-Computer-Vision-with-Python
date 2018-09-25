import numpy as np
import cv2
import math

################################################################
# hough_lines_draw
# Draw lines found in an image using Hough transform.
    
# img: Image on top of which to draw lines
# outfile: Output image filename to save plot as
# peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator
# rho: Vector of rho values, in pixels
# theta: Vector of theta values, in degrees
################################################################
def hough_lines_draw(img, outfile, peaks, rho, theta):
	height, width = img.shape
	# print "Height and Width: " + str(height) + " " + str(width)
	img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	for i in range(len(peaks)):
		# Get row index of peak and calculate intercepts
		R = rho[peaks[i][0]] # 0
		T = theta[peaks[i][1]]
		
		# print "R and T " + str(i) + ": " + str(R) + " " + str(T)
		R1 = 0
		C1 = 0
		R2 = 0
		C2 = 0
		if (T == theta[0]): # Vertical Line
			R1 = 0
			C1 = (np.round(R)).astype('int64') # C1 = C2
			R2 = height - 1
			C2 = (np.round(R)).astype('int64') # C1 = C2
			if R < 0:
				C1 = C1 * (-1)
				C2 = C2 * (-1)
		elif (T == 0): # Horizontal Line (theta == -90)
			R1 = (np.round(R)).astype('int64') # R1 = R2
			C1 = 0
			R2 = (np.round(R)).astype('int64') # R1 = R2
			C2 = width - 1
		else:
			m = (-1) * np.cos(T)/np.sin(T)
			b = R/np.sin(T)
			
			# print "m: " + str(m)
			# print "b: " + str(b)
			
			point_1_id = 0
			point_2_id = 0
			
			# Calculate 4 points
			points = [[0, (-1) * (b / m)], [b, 0], [height - 1, ((height - 1 - b) / m)], [(m * (width - 1) + b), width - 1]]
			
			print points
			
			# 4 Intercept options
			if ((points[0][1] < width) & (points[0][1] >= 0)): # (R, C) = (0, val<=w-1)
				point_1_id = 0
			elif ((points[1][0] < height) & (points[1][0] >= 0)): # (R, C) = (val<=h-1, 0)
				point_1_id = 1
			elif ((points[2][1] < width) & (points[2][1] >= 0)): # (R, C) = (h-1, 0<=val <= h-1)
				point_1_id = 2
			elif ((points[3][0] < height) & (points[3][0] >= 0)): 
				point_1_id = 3
			#else:
				#print "ERROR: Line is not on the image"

			if ((point_1_id != 0) & (points[0][1] < width) & (points[0][1] >= 0)): # (R, C) = (0, val<=w-1)
				point_1_id = 0
			elif ((point_1_id != 1) & (points[1][0] < height) & (points[1][0] >= 0)): # (R, C) = (val<=h-1, 0)
				point_1_id = 1
			elif ((point_1_id != 2) & (points[2][1] < width) & (points[2][1] >= 0)): # (R, C) = (h-1, 0<=val <= h-1)
				point_1_id = 2
			elif ((point_1_id != 3) & (points[3][0] < height) & (points[3][0] >= 0)): 
				point_1_id = 3
			#else:
				#print "ERROR: Line is not on the image"
			
			points = (np.round(points)).astype('int64')
			
			R1 = points[point_1_id][0]
			C1 = points[point_1_id][1]
			R2 = points[point_2_id][0]
			C2 = points[point_2_id][1]
			
		print "Start " + str(i) + ": " + str(R1) + " " + str(C1)
		print "End " + str(i) + ": " + str(R2) + " " + str(C2)
		print "------------------------------------------------------------------"
		
		cv2.line(img_rgb,(R1, C1),(R2, C2), (255, 0, 0), 2)

	cv2.imwrite('output/' + outfile, img_rgb)