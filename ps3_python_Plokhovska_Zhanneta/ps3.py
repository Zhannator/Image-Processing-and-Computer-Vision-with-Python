# Public libraries
import sys
import os
import numpy as np
import cv2

def svd(points_2d, points_3d):
	# M* = eigenvector of AtA with smallest eigenvalue

	# Calculate A, At, and their AtA
	num_points = len(points_2d)
	A = []
	for i in range(num_points):
		x = points_3d[i][0]
		y = points_3d[i][1]
		z = points_3d[i][2]
		u = points_2d[i][0]
		v = points_2d[i][1]
		A.append([x, y, z, 1, 0, 0, 0, 0, u*x, u*y, u*z, u])
		A.append([0, 0, 0, 0, x, y, z, 1, v*x, v*y, v*z, v])
	At = np.transpose(A)	
	AtA = np.matmul(At, A)
	
	# Find eigenvectors of AtA
	eigenvalues, eigenvectors = np.linalg.eig(AtA)
	print eigenvalues
	print eigenvectors
	
	# Find eigenvector with smallest eigenvalue
	# column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i]
	
	
def main():
	# 1: Caliberation
	# Compute projection matrix that goes from 3D to 2D image coordinates
	
	# pts2d-pic_a.txt and pts3d.txt: 20 2D and 3D points of image pic_a.png
	
	
	# 1-A: Least squares function that solves for 3x4 matrix MnormA given normalized 2D and 3D lists
	# pts2d-norm-pic_a.txt and pts3d-norm.txt: normalized test points
	# Read in normalized 2d points
	points_2d_norm = np.loadtxt('input/pts2d-norm-pic_a.txt', delimiter = ' ')
	points_3d_norm = np.loadtxt('input/pts3d-norm.txt', delimiter = ' ')
	svd(points_2d_norm, points_3d_norm)
	
		
	
	
if __name__ == "__main__":
	main()