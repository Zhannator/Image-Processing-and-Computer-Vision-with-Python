#Public libraries
import numpy as np
import math
from progressbar import ProgressBar

##################################################
# disparity_ncorr
# Returns disparity image D(y,x) 
# such that L(y,x) = R(y,x+D(y,x)) when matching
# from left (L) to right (R)
##################################################
def disparity_ncorr(L, R):
	pbar = ProgressBar()
	rows, columns = L.shape # Assuming L and R are same size
	displ_t = range(-4, 4, 1) # Displacement for template (size = 9x9)
	
	# Initialize disparity image - same size as L and R
	D = np.zeros((rows, columns), np.uint8)
	
	# For every pixel in image L
	for i in pbar(range(rows)):
		for j in range(columns):
			# rtt
			rtt = 0 # Help normalization
			# Calculate dot for pixel using template from L
			for row_t in displ_t:
				L_row = i + row_t
				for column_t in displ_t:
					L_column = j + column_t
					if ((L_row >= 0) & (L_row < rows) & (L_column >= 0) & (L_column < columns)):
						rtt = rtt + (L[L_row][L_column] * L[L_row][L_column])
				
			rtx = [0] * columns # Store dot for every pixel on epipolar line
			rxx = [0] * columns # Help normalization
			ptx = [0] * columns
			
			# For every pixel in epipolar line in R
			for k in range(columns): # row stays same = i
				# Calculate dot for pixel using template from L
				for row_t in displ_t:
					L_R_row = i + row_t
					for column_t in displ_t:
						L_column = j + column_t
						R_column = k + column_t
						# rtx
						if ((L_R_row >= 0) & (L_R_row < rows) & (L_column >= 0) & (L_column < columns) & (R_column >= 0) & (R_column < columns)):
							rtx[k] = rtx[k] + (L[L_R_row][L_column] * R[L_R_row][R_column])
						# rxx
						if ((L_R_row >= 0) & (L_R_row < rows) & (R_column >= 0) & (R_column < columns)):
							rxx[k] = rxx[k] + (R[L_R_row][R_column] * R[L_R_row][R_column])
				# Normalize
				ptx[k] = rtx[k] / math.sqrt(rxx[k] * rtt)
			# Pick best match
			pixel_index_max_dot = ptx.index(max(ptx)) # Max value when pixels match
			D[i][j] = pixel_index_max_dot - j
	
	return D