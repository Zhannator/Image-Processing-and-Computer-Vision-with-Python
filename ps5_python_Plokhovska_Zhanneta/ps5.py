# Public libraries
import sys
import os
import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt
import itertools
	
def main():
	# 1-A: LK optic flow estimation
	print "\n-----------------------1-A-----------------------" 
	# Read images
	shift0 = cv2.imread(os.path.join('input/TestSeq', 'Shift0.png'), 0) * (1.0 / 255.0) # grayscale
	shiftr2 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR2.png'), 0) * (1.0 / 255.0)  # grayscale
	shiftr5u5 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR5U5.png'), 0) * (1.0 / 255.0)  # grayscale
	shiftr10 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR10.png'), 0) * (1.0 / 255.0)  # grayscale
	shiftr20 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR20.png'), 0) * (1.0 / 255.0)  # grayscale
	shiftr40 = cv2.imread(os.path.join('input/TestSeq', 'ShiftR40.png'), 0) * (1.0 / 255.0)  # grayscale

	# Save images
	#cv2.imwrite('output/Shift0.png', shift0)
	
if __name__ == "__main__":
	main()