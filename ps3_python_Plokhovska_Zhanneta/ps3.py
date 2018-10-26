# Public libraries
import sys
import os
import numpy as np
import cv2
import math
import random

##################################################
# svd_projection
# Returns matrix M given 2D and 3D point coordinates lists
##################################################
def svd_fundamental(points_2d_a, points_2d_b):
	# M* = eigenvector of AtA with smallest eigenvalue

	# Calculate A, At, and their AtA
	num_points = len(points_2d_a)
	A = []
	for i in range(num_points):
		u_b = points_2d_b[i][0]
		v_b = points_2d_b[i][1]
		u_a = points_2d_a[i][0]
		v_a = points_2d_a[i][1]
		A.append([u_b*u_a, u_b*v_a, u_b, v_b*u_a, v_b*v_a, v_b, u_a, v_a, 1])
	
	At = np.transpose(A)
	
	AtA = np.matmul(At, A)
	
	# Find eigenvectors of AtA
	eigenvalues, eigenvectors = np.linalg.eig(AtA)
	
	# Find eigenvector with smallest eigenvalue
	# column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i]
	smallest_eigenvector = eigenvectors[:, np.argmin(eigenvalues)]
	smallest_eigenvector.shape = (3, 3) # shape m matrix into 3 x 4
	
	return smallest_eigenvector

##################################################
# calculate_residual
# Returns residual between 2 2D points
##################################################
def calculate_residual(point1, point2):
	# You can do the comparison by checking the residual
	# between the predicted location of each test point using your equation and the actual location
	# given by the 2D input data. The residual is just the distance (square root of the sum of squared
	# differences in u and v).
	return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

##################################################
# predict_3d_to_2d_point
# Returns predicted 2D point coordinates using 3D point and matrix M
##################################################
def predict_3d_to_2d_point(point_3d, M):
	point_3d = np.append(point_3d, 1) # [x, y, z, 1]
	point_3d.shape = (4, 1)
	point_2d = np.matmul(M, point_3d) # [d*x, d*y, d]
	point_2d = point_2d / point_2d[2] # [x, y, 1]
	point_2d = point_2d[0:2] # [x, y]
	point_2d.shape = (1, 2)
	return [point_2d[0][0], point_2d[0][1]]

##################################################
# svd_projection
# Returns matrix M given 2D and 3D point coordinates lists
##################################################
def svd_projection(points_2d, points_3d):
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
		A.append([x, y, z, 1, 0, 0, 0, 0, (-1)*u*x, (-1)*u*y, (-1)*u*z, (-1)*u])
		A.append([0, 0, 0, 0, x, y, z, 1, (-1)*v*x, (-1)*v*y, (-1)*v*z, (-1)*v])
	
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
	
	# 1-A: Least squares function that solves for 3x4 matrix MnormA given normalized 2D and 3D lists
	# pts2d-norm-pic_a.txt and pts3d-norm.txt: normalized test points
	# Read in normalized 2d points
	print "\n-----------------------1-A-----------------------" 
	points_2d_norm = np.loadtxt('input/pts2d-norm-pic_a.txt', delimiter = ' ')
	points_3d_norm = np.loadtxt('input/pts3d-norm.txt', delimiter = ' ')
	points_len = len(points_3d_norm)
	M = svd_projection(points_2d_norm, points_3d_norm)
	print "\nThe matrix M recovered from the normalized points (3 x 4):"
	print M
	# [u, v] projection of the last point
	point_2d_predicted = predict_3d_to_2d_point(points_3d_norm[points_len - 1], M)
	print "\nThe < u, v > projection of the last point given your M matrix:"
	print point_2d_predicted
	# Calculate residual
	print points_2d_norm[points_len - 1]
	residual = calculate_residual(points_2d_norm[points_len - 1], point_2d_predicted)
	print "\nThe residual between that projected location and the actual one given:"
	print residual
	
	# 1-B: Using 3D and 2D point lists for the image, compute camera projections matrix
	print "\n-----------------------1-B-----------------------" 
	points_2d = np.loadtxt('input/pts2d-pic_b.txt', delimiter = ' ')
	points_3d = np.loadtxt('input/pts3d.txt', delimiter = ' ')
	points_len = len(points_3d)
	average_residuals = []
	residual_min = -1
	M_min = np.zeros((3, 4))
	for k in [8, 12, 16]:
		average_residuals_temp = []
		for i in range(10): # repeat 10 times
			points_2d_temp = []
			points_3d_temp = []
			rand_indexes = random.sample(range(0, points_len), k + 4) # k random points + 4 for testing
			for index in range(k):
				points_2d_temp.append(points_2d[rand_indexes[index]])
				points_3d_temp.append(points_3d[rand_indexes[index]])
			M_temp = svd_projection(points_2d_temp, points_3d_temp)
			average_residual = 0
			for index in range(k, k + 4):
				point_2d_temp = points_2d[rand_indexes[index]]
				point_3d_temp = points_3d[rand_indexes[index]]
				point_2d_predicted = predict_3d_to_2d_point(point_3d_temp, M_temp)
				average_residual += calculate_residual(point_2d_temp, point_2d_predicted)
			average_residual = average_residual / 4
			average_residuals_temp.append(average_residual)
			if ((residual_min == -1) | (average_residual < residual_min)):
				residual_min = average_residual		
				M_min = M_temp
		average_residuals.append(average_residuals_temp)
	print "\nAverage residual for each trial of each k arranged in (3 x 10) matrix (first row for k = 8, second row for k = 12, third row for k = 16):"
	print average_residuals
	print "\nThe best M matrix (3 x 4):"
	print residual_min
	print M_min
	# Predict all 2d points
	#predicted = predict_3d_points(points_3d_norm, M) 
	
	# 1-C: Solve for the camera center in the world
	print "\n-----------------------1-C-----------------------" 
	Q_inversed = np.linalg.inv(M_min[0:3, 0:3])
	C = (-1) * np.matmul(Q_inversed, M_min[0:3, 3:4])
	C = [C[0][0], C[1][0], C[2][0]]
	print "\nThe location of the camera in real 3D world coordinates:"
	print C
	
	# 2: Fundamental Matrix Estimation
	# Estimate the mapping of points in one image to lines in another
	
	# 2-A: Use least squares function to solve for 3x3 transform F
	print "\n-----------------------2-A-----------------------" 
	points_2d_a = np.loadtxt('input/pts2d-pic_a.txt', delimiter = ' ')
	points_2d_b = np.loadtxt('input/pts2d-pic_b.txt', delimiter = ' ')
	F_full_rank = svd_fundamental(points_2d_a, points_2d_b)
	print "\nThe matrix F generated using least squares function:"
	print F_full_rank
	
	# 2-B: Reduce rank of F to 2
	print "\n-----------------------2-B-----------------------" 
	
	
	# 2-C: Use F from 2A to estimate an epipolar line lb in image b corresponding to point pa in image a: lb = F*pa
	print "\n-----------------------2-C-----------------------" 
	
	lb = []
	for pa in points_2d_a:
		lb_temp = np.matmul(F, [[pa[0]], [pa[1]], [1]])
		lb.append([lb_temp[0][0], lb_temp[1][0], lb_temp[2][0]])

	print lb
		
	la = []
	for pb in points_2d_b:
		la_temp = np.matmul(F, [[pb[0]], [pb[1]], [1]])
		la.append([la_temp[0][0], la_temp[1][0], la_temp[2][0]])		
	
	print la
	
if __name__ == "__main__":
	main()