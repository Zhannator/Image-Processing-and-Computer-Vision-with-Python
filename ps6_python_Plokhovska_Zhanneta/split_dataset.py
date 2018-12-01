# Public libraries
import os
import shutil
import sys
import cv2
import numpy as np
import math
import random
import itertools
from sklearn.neighbors import KNeighborsClassifier

def clear_folder(folder):
	for file in os.listdir(folder):
		path = os.path.join(folder, file)
		try:
			if os.path.isfile(path):
				os.unlink(path)
			elif os.path.isdir(path):
				shutil.rmtree(path)
		except Exception as e:
			print(e)

def main():

	# ----------------------------------------------
	# SPLIT DATA INTO TRAINING AND TESTING SETS
	# ----------------------------------------------
	
	print "\nCreate Training and testing data..."	

	num_of_people = 40
	num_of_faces_per_person = 10
	training_images_per_class = num_of_faces_per_person * 0.8
	testing_images_per_class = num_of_faces_per_person * 0.2
	all_folder = 'input/all/'
	training_folder = 'input/train/'
	testing_folder = 'input/test/'
	
	# Clear training and testing directories
	clear_folder(training_folder)
	clear_folder(testing_folder)
	
	# Iterate through facial images and separate them into training and testing matrixes (80 : 20 ratio) and save them under traina and test directories in input folder
	train_filename = 1
	for i in range(1, num_of_people + 1, 1):
		rand_indexes = random.sample(range(1, num_of_faces_per_person + 1), 10)
		for j in range(training_images_per_class):
			# Read image
			img = cv2.imread(all_folder + 's' + str(i) + '/' + str(rand_indexes[j]) + '.pgm', 0)
			cv2.imwrite(training_folder + str(train_filename) + '.pgm', img)
			train_filename = train_filename + 1
		
		os.mkdir(testing_folder + 's' + str(i))
		test_filename = 1
		for j in range(testing_images_per_class):
			# Read image
			img = cv2.imread(all_folder + 's' + str(i) + '/' + str(rand_indexes[j + training_images_per_class]) + '.pgm', 0)
			cv2.imwrite(testing_folder + 's' + str(i) + '/' + str(test_filename) + '.pgm', img)
			test_filename = test_filename + 1
		
if __name__ == "__main__":
	main()