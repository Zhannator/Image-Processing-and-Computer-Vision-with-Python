# Public Libraries
import sys
import cv2
import numpy as np
#from PIL import Image
from matplotlib import pyplot as plt

# Private Libraries
from hough_lines_acc import hough_lines_acc
from hough_peaks import hough_peaks
from hough_lines_draw import hough_lines_draw
#import hough_circles_acc
#import find_circles

def main():
	if 1 == 0:
		# 1-A: Canny Edge Detection
		# Open input images
		img = cv2.imread('input/ps1-input0.png', 0)
		# Compute edge image img_edges
		img_edges = cv2.Canny(img, 100, 200)
		cv2.imwrite('output/ps1-1-a-1.png', img_edges)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(img, cmap = 'gray')
				plt.title('Original Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_edges, cmap = 'gray')
				plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
				plt.show()
				
		# 2-A: Hough Lines Accumulator
		H, theta, rho = hough_lines_acc(img_edges) #{H, theta, rho}
		# Normalize hough_lines_acc_output['H'] 
		hough_normalized = H.astype('uint8')
		# Plot/show accumulator array H, save as output/ps1-2-a-1.png
		cv2.imwrite('output/ps1-2-a-1.png', hough_normalized)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(img_edges, cmap = 'gray')
				plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(H, cmap = 'gray')
				plt.title('Hough Image'), plt.xticks([]), plt.yticks([])
				plt.show()
				
		# 2-B: Hough Peaks
		hough_highlighted_peaks = hough_normalized.copy()
		num_of_peaks = 10
		peaks = hough_peaks(H, num_of_peaks)
		# Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png
		for i in range(len(peaks)):
			cv2.circle(hough_highlighted_peaks, (peaks[i][1], peaks[i][0]), 10, (255, 255, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
		# Plot/show accumulator array H, save as output/ps1-2-a-1.png
		cv2.imwrite('output/ps1-2-b-1.png', hough_highlighted_peaks)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(H, cmap = 'gray')
				plt.title('Hough Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(hough_highlighted_peaks, cmap = 'gray')
				plt.title('Peaks Image'), plt.xticks([]), plt.yticks([])
				plt.show()
		
		# 2-C: Draw green lines on original image
		hough_lines_draw(img, 'ps1-2-c-1.png', peaks, rho, theta)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				hough_lines = cv2.imread('output/ps1-2-c-1.png', 0)
				plt.subplot(121),plt.imshow(hough_highlighted_peaks, cmap = 'gray')
				plt.title('Hough Peaks'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(hough_lines, cmap = 'gray')
				plt.title('Lines Image'), plt.xticks([]), plt.yticks([])
				plt.show()	
	
		# 3-A: Gaussian Blur
		img_noise = cv2.imread('input/ps1-input0-noise.png', 0)
		# Apply Gaussian Blur
		img_noise_blur = cv2.GaussianBlur(img_noise.copy(), (5, 5), 30)
		img_noise_blur = img_noise_blur.astype('uint8')
		cv2.imwrite('output/ps1-3-a-1.png', img_noise_blur)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(img_noise, cmap = 'gray')
				plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_noise_blur, cmap = 'gray')
				plt.title('Smoothed Image'), plt.xticks([]), plt.yticks([])
				plt.show()
		
		# 3-B: Edge Detection (Noisy vs. Smoothed)
		# Compute edge image img_edges
		img_noise_edges = cv2.Canny(img_noise, 155, 200)
		img_noise_blur_edges = cv2.Canny(img_noise_blur, 155, 200)
		cv2.imwrite('output/ps1-3-b-1.png', img_noise_edges)
		cv2.imwrite('output/ps1-3-b-2.png', img_noise_blur_edges)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(img_noise_edges, cmap = 'gray')
				plt.title('Noisy Image - Edge'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_noise_blur_edges, cmap = 'gray')
				plt.title('Smoothed Image - Edge'), plt.xticks([]), plt.yticks([])
				plt.show()
		
		# 3-C: Hough Transform on Smoothed Image
		H, theta, rho = hough_lines_acc(img_noise_blur_edges) #{H, theta, rho}
		# Normalize hough_lines_acc_output['H'] 
		hough_normalized = H.astype('uint8')
		num_of_peaks = 10
		peaks = hough_peaks(H, num_of_peaks, [18, 5]) # Increased nHoodSize to get rid of extra circles in the same neighborhood
		# Highlight peak locations on accumulator array
		for i in range(len(peaks)):
			cv2.circle(hough_normalized, (peaks[i][1], peaks[i][0]), 10, (255, 255, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
		# Save hough image with highlighted circles
		cv2.imwrite('output/ps1-3-c-1.png', hough_normalized)
		# Draw lines
		hough_lines_draw(img_noise, 'ps1-3-c-2.png', peaks, rho, theta)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				hough_lines = cv2.imread('output/ps1-3-c-2.png', 0)
				plt.subplot(121),plt.imshow(img_noise, cmap = 'gray')
				plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(hough_lines, cmap = 'gray')
				plt.title('Hough Image'), plt.xticks([]), plt.yticks([])
				plt.show()
	
	# 4-A: Blur monochrome image that contains coins and pens
	# Read in an image with coins and pens
	img_circles_and_lines = cv2.imread('input/ps1-input1.png', cv2.IMREAD_GRAYSCALE)
	# Apply Gaussian Blur
	img_circles_and_lines_blur = cv2.GaussianBlur(img_circles_and_lines.copy(), (3, 3), 3)
	img_circles_and_lines_blur = img_circles_and_lines_blur.astype('uint8')
	# Save smoothed monocrome image
	cv2.imwrite('output/ps1-4-a-1.png', img_circles_and_lines_blur)
	# Optional plot/show
	if (len(sys.argv) > 1):
		if (sys.argv[1] == "-plot"):
			plt.subplot(121),plt.imshow(img_circles_and_lines, cmap = 'gray')
			plt.title('Monochrome Image'), plt.xticks([]), plt.yticks([])
			plt.subplot(122),plt.imshow(img_circles_and_lines_blur, cmap = 'gray')
			plt.title('Smoothed Monochrome Image'), plt.xticks([]), plt.yticks([])
			plt.show()
			
	# 4-B: Find edges in smoothed monochrome image that contains coins and pens
	img_circles_and_lines_blur_edges = cv2.Canny(img_circles_and_lines_blur, 100, 200)
	cv2.imwrite('output/ps1-4-b-1.png', img_circles_and_lines_blur_edges)
	if (len(sys.argv) > 1):
		if (sys.argv[1] == "-plot"):
			plt.subplot(121),plt.imshow(img_circles_and_lines_blur, cmap = 'gray')
			plt.title('Smoothed Monochrome Image'), plt.xticks([]), plt.yticks([])
			plt.subplot(122),plt.imshow(img_circles_and_lines_blur_edges, cmap = 'gray')
			plt.title('Smoothed Monochrome Image - Edges'), plt.xticks([]), plt.yticks([])
			plt.show()	

	# 4-C: Hough transform on Image with Coins and Pens
	H, theta, rho = hough_lines_acc(img_circles_and_lines_blur_edges) #{H, theta, rho}
	# Normalize hough_lines_acc_output['H'] 
	hough_normalized = H.astype('uint8')
	num_of_peaks = 10
	peaks = hough_peaks(H, num_of_peaks)
	print peaks
	# Highlight peak locations on accumulator array
	for i in range(len(peaks)):
		cv2.circle(hough_normalized, (peaks[i][1], peaks[i][0]), 10, (255, 255, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
	# Save hough image with highlighted circles
	cv2.imwrite('output/ps1-4-c-1.png', hough_normalized)
	# Draw lines
	hough_lines_draw(img_circles_and_lines, 'ps1-4-c-2.png', peaks, rho, theta)
	# Optional plot/show
	if (len(sys.argv) > 1):
		if (sys.argv[1] == "-plot"):
			hough_lines = cv2.imread('output/ps1-4-c-2.png', 0)
			plt.subplot(121),plt.imshow(img_circles_and_lines, cmap = 'gray')
			plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
			plt.subplot(122),plt.imshow(hough_lines, cmap = 'gray')
			plt.title('Hough Image'), plt.xticks([]), plt.yticks([])
			plt.show()
	
if __name__ == "__main__":
	main()