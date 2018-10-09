import numpy as np
import math
from progressbar import ProgressBar

##################################################
# disparity_ssd
# Returns disparity image D(y,x) 
# such that L(y,x) = R(y,x+D(y,x)) when matching
# from left (L) to right (R)
##################################################
def disparity_ssd(L, R):
	pbar = ProgressBar()
	rows, columns = L.shape # Assuming L and R are same size
	displ_t = range(-3, 3, 1) # Displacement for template (size = 7x7)
	
	# Initialize disparity image - same size as L and R
	D = np.zeros((rows, columns), np.uint8)
	
	# For every pixel in image L
	for i in pbar(range(rows)):
		for j in range(columns):
			pixel_ssd = [0] * columns # Store ssd for every pixel on epipolar line
			# For every pixel in epipolar line in R
			for k in range(columns): # row stays same = i
				# Calculate ssd for pixel using template from L
				for row_t in displ_t:
					L_R_row = i + row_t
					for column_t in displ_t:
						L_column = j + column_t
						R_column = k + column_t
						if ((L_R_row >= 0) & (L_R_row < rows) & (L_column >= 0) & (L_column < columns) & (R_column >= 0) & (R_column < columns)):
							pixel_ssd[k] = pixel_ssd[k] + math.pow(L[L_R_row][L_column] - R[L_R_row][R_column], 2)
			# Pick best match
			pixel_index_min_ssd = pixel_ssd.index(min(pixel_ssd))
			D[i][pixel_index_min_ssd] = min(pixel_ssd)
	return D