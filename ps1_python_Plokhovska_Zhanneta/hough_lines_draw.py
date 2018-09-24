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
	print "Height and Width: " + str(height) + " " + str(width)
	img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	for i in range(len(peaks)):
		# Get row index of peak and calculate intercepts
		R = rho[peaks[i][0]] # 0
		T = theta[peaks[i][1]]
		
		print "R and T " + str(i) + ": " + str(R) + " " + str(T)
		
		R1 = 0
		C1 = 0
		R2 = 0
		C2 = 0
		if (T == theta[0]): # Vertical Line
			R1 = 0
			C1 = np.round(R) # C1 = C2
			R2 = height - 1
			C2 = np.round(R) # C1 = C2
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
			print "m: " + str(m)
			print "b: " + str(b)
			# 6 possibilities: (0,0), (h-1, 0), (0, w-1), (h-1, w-1), (value < height, 0), (0, value < width)
			if (((-1) * b / m <= width - 1) & ((-1) * b / m >= 0)): # (R, C) = (0, val<=w-1)
				R1 = 0
				C1 = (-1) * b / m
				print "A: " + str(R1) + " " + str(C1)
			elif ((b <= height - 1) & (b >= 0)): # (R, C) = (val<=h-1, 0)
				R1 = b
				C1 = 0
				print "B: " + str(R1) + " " + str(C1)
			elif ((((height - 1 - b) / m) <= width - 1) & (((height - 1 - b) / m) >= 0)): # (R, C) = (h-1, 0<=val <= h-1)
				R1 = height - 1
				C1 = ((height - 1 - b) / m)
				print "C: " + str(R1) + " " + str(C1)
			else: 
				R1 = m * (width - 1) + b
				C1 = width - 1
				print "D: " + str(R1) + " " + str(C1)
				
			if ((R1 != 0) & (C1 != (-1) * b / m) & ((-1) * b / m <= width - 1) & (((-1) * b / m >= 0))): # (R, C) = (0, val<=w-1)
				R2 = 0
				C2 = (-1) * b / m
				print "E: " + str(R2) + " " + str(C2)
			elif ((R1 != b) & (C1 != 0) & (b <= height - 1) & (b >= 0)): # (R, C) = (val<=h-1, 0)
				R2 = b
				C2 = 0
				print "F: " + str(R2) + " " + str(C2)
			elif ((R1 != height - 1) & (C1 != ((height - 1 - b) / m)) & (((height - 1 - b) / m) <= width - 1) & (((height - 1 - b) / m) >= 0)): # (R, C) = (h-1, 0<=val <= h-1)
				R2 = height - 1
				C2 = ((height - 1 - b) / m)
				print "G: " + str(R2) + " " + str(C2)
			else: 
				R2 = m * (width - 1) + b
				C2 = width - 1
				print "H: " + str(R2) + " " + str(C2)
			
		print "Start " + str(i) + ": " + str(R1) + " " + str(C1)
		print "End " + str(i) + ": " + str(R2) + " " + str(C2)
		print "------------------------------------------------------------------"
		cv2.line(img_rgb,(int(R1), int(C1)),(int(R2), int(C2)),(255, 0, 0),2)
		#cv2.line(img_rgb,(C1, R1),(C2, R2),(255, 0, 0),2)

	cv2.imwrite('output/' + outfile, img_rgb)