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
	
	# Find eigenvector with smallest eigenvalue
	# column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i]
	smallest_eigenvector = eigenvectors[:, np.argmin(eigenvalues)]
	smallest_eigenvector.shape = (3, 4) # shape m matrix into 3 x 4
	
	return smallest_eigenvector
	
def main():
	# 1: Caliberation
	# Compute projection matrix that goes from 3D to 2D image coordinates
	
	# pts2d-pic_a.txt and pts3d.txt: 20 2D and 3D points of image pic_a.png
	
	
	# 1-A: Least squares function that solves for 3x4 matrix MnormA given normalized 2D and 3D lists
	# pts2d-norm-pic_a.txt and pts3d-norm.txt: normalized test points
	# Read in normalized 2d points
	points_2d_norm = np.loadtxt('input/pts2d-norm-pic_a.txt', delimiter = ' ')
	points_3d_norm = np.loadtxt('input/pts3d-norm.txt', delimiter = ' ')
	points_3d_norm_len = len(points_3d_norm)
	M = svd(points_2d_norm, points_3d_norm)
	# [u, v] projection of the last point
	point_3d_last = points_3d_norm[points_3d_norm_len - 1] # [x, y, z]
	point_3d_last = np.append(point_3d_last, 1) # [x, y, z, 1]
	point_3d_last.shape = (4, 1)
	print "\nM: \n"
	print M
	print "\n3D Point: \n"
	print point_3d_last
	point_2d_calculated = np.matmul(M, point_3d_last)
	point_2d_calculated = point_2d_calculated / point_2d_calculated[2]
	point_2d_calculated = point_2d_calculated[0:2]
	print "\n2D Point: \n"
	print point_2d_calculated
	
	
if __name__ == "__main__":
	main()