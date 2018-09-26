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
from hough_circles_acc import hough_circles_acc
from find_circles import find_circles

def main():
	if 1 == 0:
	
		# ---------------------------------------------------------- 1 ----------------------------------------------------------

		# 1-A: Canny Edge Detection
		print "--------------1-A--------------"
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

		# ---------------------------------------------------------- 2 ----------------------------------------------------------

				
		# 2-A: Hough Lines Accumulator
		print "--------------2-A--------------"
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
		print "--------------2-B--------------"
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
		print "--------------2-C--------------"
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

		# ---------------------------------------------------------- 3 ----------------------------------------------------------
	   
		# 3-A: Gaussian Blur
		print "--------------3-A--------------"
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
		print "--------------3-B--------------"
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
		print "--------------3-C--------------"
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
				
		# ---------------------------------------------------------- 4 ----------------------------------------------------------
		
		# 4-A: Blur monochrome image that contains coins and pens
		print "--------------4-A--------------"
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
		print "--------------4-B--------------"
		img_circles_and_lines_blur_edges = cv2.Canny(img_circles_and_lines_blur, 300, 400)
		cv2.imwrite('output/ps1-4-b-1.png', img_circles_and_lines_blur_edges)
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(img_circles_and_lines_blur, cmap = 'gray')
				plt.title('Smoothed Monochrome Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_circles_and_lines_blur_edges, cmap = 'gray')
				plt.title('Smoothed Monochrome Image - Edges'), plt.xticks([]), plt.yticks([])
				plt.show()	

		# 4-C: Hough transform on Image with Coins and Pens
		print "--------------4-C--------------"
		H, theta, rho = hough_lines_acc(img_circles_and_lines_blur_edges) #{H, theta, rho}
		# Normalize hough_lines_acc_output['H'] 
		hough_normalized = H.astype('uint8')
		num_of_peaks = 10
		peaks = hough_peaks(hough_normalized, num_of_peaks, [30, 30], 60)
		#print "Peaks " + str(peaks)
		# Highlight peak locations on accumulator array
		for i in range(len(peaks)):
			cv2.circle(hough_normalized, (peaks[i][1], peaks[i][0]), 10, (255, 255, 255), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
		# Save hough image with highlighted circles
		cv2.imwrite('output/ps1-4-c-1.png', hough_normalized)
		# Draw lines
		cv2.imwrite('output/ps1-4-c-2.png', img_circles_and_lines)
		hough_lines_draw(img_circles_and_lines, 'ps1-4-c-2.png', peaks, rho, theta)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				hough_lines = cv2.imread('output/ps1-4-c-2.png', 0)
				plt.subplot(121),plt.imshow(img_circles_and_lines, cmap = 'gray')
				plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(hough_lines, cmap = 'gray')
				plt.title('Lines Image'), plt.xticks([]), plt.yticks([])
				plt.show()
		
		# ---------------------------------------------------------- 5 ----------------------------------------------------------
		
		# 5-A: Compute accumulator array for given radius
		print "--------------5-A--------------"
		# Save smoothed monocrome image and edge image
		cv2.imwrite('output/ps1-5-a-1.png', img_circles_and_lines_blur)
		cv2.imwrite('output/ps1-5-a-2.png', img_circles_and_lines_blur_edges)
		# Smooth monochrome image - edges: img_circles_and_lines_blur_edges
		radius = 20
		H = hough_circles_acc(img_circles_and_lines_blur_edges, radius)
		# Find circle centers
		num_of_circles = 10
		centers = hough_peaks(H, num_of_circles, [30, 30], 85);
		# Draw circles
		original_plus_circles  = img_circles_and_lines.copy()
		img_rgb = cv2.cvtColor(original_plus_circles,cv2.COLOR_GRAY2RGB)
		for i in range(len(centers)):
			cv2.circle(img_rgb, (centers[i][1], centers[i][0]), radius, (255, 0, 0), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
		# Save hough image with highlighted circles
		cv2.imwrite('output/ps1-5-a-3.png', img_rgb)
			# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(H, cmap = 'gray')
				plt.title('Hough Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_rgb, cmap = 'gray')
				plt.title('Original with Some Highlighted Circles'), plt.xticks([]), plt.yticks([])
				plt.show()
			
		# 5-B: Combine 5-A into one function call to find_circles
		print "--------------5-B--------------"
		min_radius = 20
		max_radius = 50
		centers, radii = find_circles(img_circles_and_lines_blur_edges, [min_radius, max_radius])
		# Draw circles
		original_plus_circles  = img_circles_and_lines.copy()
		img_rgb = cv2.cvtColor(original_plus_circles,cv2.COLOR_GRAY2RGB)
		for i in range(len(centers)):
			cv2.circle(img_rgb, (centers[i][1], centers[i][0]), radii[i], (255, 0, 0), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
		# Save hough image with highlighted circles
		cv2.imwrite('output/ps1-5-b-1.png', img_rgb)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(img_circles_and_lines, cmap = 'gray')
				plt.title('Original Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_rgb, cmap = 'gray')
				plt.title('Original with All Highlighted Circles'), plt.xticks([]), plt.yticks([])
				plt.show()	
	
	if 1 == 1:
		# ---------------------------------------------------------- 6 ----------------------------------------------------------
	
		# 6-A: Line finder on realistic images
		print "--------------6-A--------------"
		# Read in an image
		img = cv2.imread('input/ps1-input2.png', cv2.IMREAD_GRAYSCALE)
		# Apply Gaussian Blur
		img_blur = cv2.GaussianBlur(img.copy(), (3, 3), 3)
		img_blur = img_blur.astype('uint8')
		img_blur_edges = cv2.Canny(img_blur, 60, 220)
		H, theta, rho = hough_lines_acc(img_blur_edges) #{H, theta, rho}
		# Normalize hough_lines_acc_output['H'] 
		hough_normalized = H.astype('uint8')
		num_of_peaks = 10
		peaks = hough_peaks(H, num_of_peaks, [30, 30], 70);
		# Draw lines
		hough_lines_draw(img_blur, 'ps1-6-a-1.png', peaks, rho, theta)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				img_pens = cv2.imread('output/ps1-6-a-1.png', cv2.IMREAD_GRAYSCALE)
				plt.subplot(121),plt.imshow(img, cmap = 'gray')
				plt.title('Original Busy Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_pens, cmap = 'gray')
				plt.title('Busy Image with Highlighted Pens'), plt.xticks([]), plt.yticks([])
				plt.show()	
	
		# 6-C: Attempt to find lines that are just pen boundaries
		print "--------------6-C--------------"
		# Read in an image
		img = cv2.imread('input/ps1-input2.png', cv2.IMREAD_GRAYSCALE)
		# Apply Gaussian Blur
		img_blur = cv2.GaussianBlur(img.copy(), (5, 5), 5)
		img_blur = img_blur.astype('uint8')
		
		img_blur_edges = cv2.Canny(img_blur, 100, 200)
		H, theta, rho = hough_lines_acc(img_blur_edges) #{H, theta, rho}
		# Normalize hough_lines_acc_output['H'] 
		hough_normalized = H.astype('uint8')
		
		# Save image
		cv2.imwrite('output/z.png', hough_normalized)
		
		num_of_peaks = 10
		peaks = hough_peaks(H, num_of_peaks, [30, 30], 120);
		# Draw lines
		hough_lines_draw(img_blur, 'ps1-6-c-1.png', peaks, rho, theta)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				img_pens = cv2.imread('output/ps1-6-c-1.png', cv2.IMREAD_GRAYSCALE)
				plt.subplot(121),plt.imshow(img, cmap = 'gray')
				plt.title('Original Busy Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_pens, cmap = 'gray')
				plt.title('Busy Image with Highlighted Pens'), plt.xticks([]), plt.yticks([])
				plt.show()	
			
		# 7-A: Finding Circles on the same clutter image
		print "--------------7-A--------------"		
		# Read in an image
		img = cv2.imread('input/ps1-input2.png', cv2.IMREAD_GRAYSCALE)
		# Apply Gaussian Blur
		img_blur = cv2.GaussianBlur(img.copy(), (5, 5), 10)
		img_blur = img_blur.astype('uint8')	
		img_blur_edges = cv2.Canny(img_blur, 60, 220)
		min_radius = 20
		max_radius = 50
		centers, radii = find_circles(img_blur_edges, [min_radius, max_radius])
		# Draw circles
		original_plus_circles  = img.copy()
		img_rgb = cv2.cvtColor(original_plus_circles,cv2.COLOR_GRAY2RGB)
		for i in range(len(centers)):
			cv2.circle(img_rgb, (centers[i][1], centers[i][0]), radii[i], (255, 0, 0), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
		# Save hough image with highlighted circles
		cv2.imwrite('output/ps1-7-a-1.png.png', img_rgb)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(img, cmap = 'gray')
				plt.title('Original Busy Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_rgb, cmap = 'gray')
				plt.title('Busy Image with Highlighted Circles'), plt.xticks([]), plt.yticks([])
				plt.show()	
	
	if 1 == 0:
		# 8-A: Line and circle finders on distorted image
		print "--------------8-A--------------"	
		# Read in an image
		img = cv2.imread('input/ps1-input3.png', cv2.IMREAD_GRAYSCALE)
		
		# Apply Gaussian Blur
		img_blur = cv2.GaussianBlur(img.copy(), (5, 5), 5)
		img_blur = img_blur.astype('uint8')
		img_blur_edges = cv2.Canny(img_blur, 80, 220)
		# -- Lines --
		H, theta, rho = hough_lines_acc(img_blur_edges) #{H, theta, rho}
		# Normalize hough_lines_acc_output['H'] 
		hough_normalized = H.astype('uint8')
		num_of_peaks = 10
		peaks = hough_peaks(H, num_of_peaks, [30, 30], 70);
		# Draw lines
		hough_lines_draw(img_blur, 'ps1-8-a-1.png', peaks, rho, theta)
		# Read in an image
		img = cv2.imread('output/ps1-8-a-1.png', cv2.IMREAD_GRAYSCALE)
		# -- Circles --
		min_radius = 20
		max_radius = 50
		centers, radii = find_circles(img_blur_edges, [min_radius, max_radius])
		# Draw circles
		original_plus_circles  = cv2.imread('output/ps1-8-a-1.png', cv2.IMREAD_GRAYSCALE) # Read the image back in
		img_rgb = cv2.cvtColor(original_plus_circles,cv2.COLOR_GRAY2RGB)
		for i in range(len(centers)):
			cv2.circle(img_rgb, (centers[i][1], centers[i][0]), radii[i], (255, 0, 0), thickness=2, lineType=8) # Draw a circle using center coordinates and radius
		# Save hough image with highlighted circles
		cv2.imwrite('output/ps1-8-a-1.png', img_rgb)
		# Optional plot/show
		if (len(sys.argv) > 1):
			if (sys.argv[1] == "-plot"):
				plt.subplot(121),plt.imshow(img, cmap = 'gray')
				plt.title('Original Distorted Image'), plt.xticks([]), plt.yticks([])
				plt.subplot(122),plt.imshow(img_rgb, cmap = 'gray')
				plt.title('Distorted Image with Highlighted Circles/Lines'), plt.xticks([]), plt.yticks([])
				plt.show()	
		
if __name__ == "__main__":
	main()