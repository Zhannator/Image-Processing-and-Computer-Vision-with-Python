import numpy as np
import cv2
import math

def pol2cartesian(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return (x, y)

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
	img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	for i in range(len(peaks)):
		# Get row index of peak and calculate intercepts
		R = rho[peaks[i][0]] # 0
		T = theta[peaks[i][1]]
		
		m = -np.cos(T)/np.sin(T)
		b = R/np.sin(T)
		
		print "R and T " + str(i) + ": " + str(R) + " " + str(T)		
		C1 = 0
		R1 = 0
		C2 = 0
		R2 = 0
		if (T == 0): # Vertical Line
			R1 = 0
			C1 = (np.round(R)).astype('int64') # C1 = C2
			R2 = height - 1
			C2 = (np.round(R)).astype('int64') # C1 = C2
		elif (T == theta[0]): # Horizontal Line (theta == -90)
			R1 = (np.round(R)).astype('int64') # R1 = R2
			C1 = 0
			R2 = (np.round(R)).astype('int64') # R1 = R2
			C2 = width - 1
		else:
			R1 = 0
			C1 = (np.round(R/np.sin(T))).astype('int64') # when x = 0
			R2 = (np.round(R/np.cos(T))).astype('int64') # when y = 0
			C2 = 0
			
		print "Start " + str(i) + ": " + str(R1) + " " + str(C1)
		print "End " + str(i) + ": " + str(R2) + " " + str(C2)
		
		cv2.line(img_rgb,(R1, C1),(R2, C2),(255, 0, 0),2)
		#cv2.line(img_rgb,(C1, R1),(C2, R2),(255, 0, 0),2)

	cv2.imwrite('output/' + outfile, img_rgb)