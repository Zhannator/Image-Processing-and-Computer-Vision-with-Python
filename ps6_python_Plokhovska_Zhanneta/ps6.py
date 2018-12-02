# Public libraries
import sys
import cv2
import numpy as np
import math
import random
import itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

################################################################################
# Calculate eigenfaces and their according eigenvalues
################################################################################
def pca_analysis(T):
	# Calculate mean face
	m = (T.mean(1))
	print m.shape
	
	# Subtract mean from each column vector in (a.k.a. center images)
	A = np.transpose(np.transpose(T) - m)
	
	#print "\nA Shape: {}\n".format(A.shape)
	
	# Calculate convergence matrix
	#C = np.matmul(A, np.transpose(A))
	
	# Calculate convergence matrix
	AtA = np.matmul(np.transpose(A), A)
	
	#print "\nAtA Shape: {}\n".format(AtA.shape)
	
	# Calculate eigenvalues and eigenfaces of AtA
	eigenvalues, eigenfaces = np.linalg.eig(AtA)
	
	# Sort eigenvalues and eigenvectors (largest to smallest)
	idx = eigenvalues.argsort()[::-1]   
	eigenvalues = eigenvalues[idx]
	eigenfaces = eigenfaces[:,idx]	
	
	# Calculate U (most important eigenvectors from AAt) by multiplying A and V (only most important eigenvectors)
	U = np.matmul(A, eigenfaces)
	
	return eigenvalues, U, AtA

################################################################################
# Extract and return features from each image in images (a.k.a. Projection)
################################################################################
def pca_extract_features(U, images, m):
	U_T = np.transpose(U)
	W_training = []
	images = np.transpose(images)
	num_images = len(images)
	images_unique = images - m
	for i in range(num_images):
		W_training.append(np.dot(U_T, images_unique[i]))
	return np.array(W_training)

################################################################################
# Normalize data using mean and standard deviation
################################################################################	
def pca_normalize(data):
	# Column-wise subtract the mean and divide by the std deviation
	rows, columns = data.shape
	for r in range(rows):
		data[r] = (data[r] - (data[r]).mean(0)) / np.std(data[r])
		
	return data

################################################################################
# Calculate minimum number of eigenvectors needed to capture min_variance
################################################################################
def reduce_number_of_eigenvectors(eigenvalues_training, min_variance):
	eigenvalues_training_len = len(eigenvalues_training)
	eigenvalues_training_sum = np.sum(eigenvalues_training)
	k_min_variance = 0
	v_list = np.zeros(eigenvalues_training_len)
	for k in range(eigenvalues_training_len):
		v = np.sum(eigenvalues_training[0:k]) / eigenvalues_training_sum
		v_list[k] = v
	
	for k in range(eigenvalues_training_len):
		if v_list[k] >= min_variance:
			return (k + 1), v_list # Add one because k count starts at 0
			
#################################################################################
# Normalize image for display
#################################################################################
def normalize_for_display(img, distribute = True):
	img = img - img.min() # Get rid of negative values
	if distribute == True:
		img = (img * (1/img.max()) * 255) # distribute values between 0 and 255
	return img.astype('uint8')
			
def main():
	# Helpful constants
	img_height = 112
	img_width = 92
	num_of_people = 40
	num_of_faces_per_person = 10
	training_images_per_class = int(num_of_faces_per_person * 0.8)
	testing_images_per_class = int(num_of_faces_per_person * 0.2)
	num_of_pixels = img_height * img_width
	training_images_total = num_of_people * training_images_per_class
	testing_images_total = num_of_people * testing_images_per_class
	training_folder = 'input/train/'
	testing_folder = 'input/test/'

	# 1-A: Training images
	print "\n-----------------------1-A-----------------------" 
	# Training images
	training_images = np.zeros((training_images_total, num_of_pixels)) # training images, each as row vector
	training_classes = np.zeros(training_images_total) # training classes
	# Testing images
	testing_images = np.zeros((testing_images_total, num_of_pixels)) # testing images, each as row vector
	testing_classes = np.zeros(testing_images_total) # testing classes
	# Read in training images
	counter = 0
	for i in range(1, training_images_total + 1, 1):
		# Read image
		img = cv2.imread(training_folder + str(i) + ".pgm", 0) * (1.0 / 255.0)
		# 2D image matrix to 1D row vector
		training_images[counter] = (img).flatten()
		training_classes[counter] = int(counter / training_images_per_class) + 1
		counter = counter + 1	
	# Read in testing images
	counter = 0
	for i in range(1, num_of_people + 1, 1):
		for j in range(1, testing_images_per_class + 1, 1):
			# Read image
			img = cv2.imread(testing_folder + 's' + str(i) + '/' + str(j) + '.pgm', 0) * (1.0 / 255.0)
			# 2D image matrix to 1D row vector
			testing_images[counter] = (img).flatten()
			testing_classes[counter] = i
			counter = counter + 1
	# Row to column images
	T =  np.transpose(training_images)
	T_testing = np.transpose(testing_images)
	cv2.imwrite('output/ps6-1-a.png', normalize_for_display(T))
	# Convert images to type double
	T = T.astype(np.double)
	T_testing = T_testing.astype(np.double)
	
	# 1-B: Average face
	print "\n-----------------------1-B-----------------------" 
	# Calculate mean face
	m = T.mean(1)
	cv2.imwrite('output/ps6-1-b.png', normalize_for_display(np.reshape(m, (img_height, img_width))))
	
	# 1-C: PCA analysis
	print "\n-----------------------1-C-----------------------" 
	eigenvalues, eigenfaces, C = pca_analysis(T)
	cv2.imwrite('output/ps6-1-c-1.png', normalize_for_display(C))
	# First 8 eigenfaces into 1 figure
	eight_eigenfaces = np.reshape(eigenfaces[:, 0], (img_height, img_width))
	for i in range(1, 8, 1):
		eigenface = np.reshape(eigenfaces[:, i], (img_height, img_width))
		eight_eigenfaces = cv2.hconcat([eight_eigenfaces, eigenface])
	cv2.imwrite('output/ps6-1-c-2.png', normalize_for_display(eight_eigenfaces))

	# 1-D: Capture 95% of variance
	print "\n-----------------------1-D-----------------------" 
	# Decide how many eigenfaces are enough to represent variance in our training set - at least 95 % variance
	k, v_list = reduce_number_of_eigenvectors(eigenvalues, 0.95)
	print "\nk = {}\n".format(k)
	# Plot v_list vs k
	plt.plot(range(1, len(v_list) + 1, 1), v_list)
	plt.xlabel('k')
	plt.ylabel('v(k)')
	#plt.show()
	plt.savefig('output/ps6-1-d-1.png')
	# Dominant eigenvectors
	U = eigenfaces[:, 0 : k]
	
	# 2-A: Feature extraction -training
	print "\n-----------------------2-A-----------------------" 
	W_training = pca_extract_features(U, T, m)
	# Normalize data
	W_training = pca_normalize(W_training)	
	print "\n W training dimensions: {}".format(W_training.shape)

	# 2-B: Feature extraction - testing
	print "\n-----------------------2-B-----------------------" 
	W_testing = pca_extract_features(U, T_testing, m)
	print "\n W testing dimensions: {}".format(W_testing.shape)
	# Normalize data	
	W_testing = pca_normalize(W_testing)
	
	# 3-A: KNN
	print "\n-----------------------3-A-----------------------" 
	num_of_neighbors = np.array([1, 3, 5, 7, 9, 11])
	accuracies = np.zeros(len(num_of_neighbors))
	# Test for accuracy using sklearn algorithm
	for i in range(len(num_of_neighbors)):
		knn = KNeighborsClassifier(n_neighbors = num_of_neighbors[i])
		knn.fit(W_training, training_classes)
		pred = knn.predict(W_testing)
		rows_testing, cols_testing = W_testing.shape
		total_correct = 0.0
		total_incorrect = 0.0	
		for r in range(rows_testing):
			# Check if the classification is correct
			if pred[r] == testing_classes[r]:
				total_correct = total_correct + 1.0
			else:
				total_incorrect = total_incorrect + 1.0
		accuracies[i] = ((total_correct / (total_correct + total_incorrect)) * 100) # Accuracy
	print "\nNumber of neighbors (k): {}\n".format(num_of_neighbors)
	print "\nAccuracies: {}\n".format(accuracies)
	
if __name__ == "__main__":
	main()