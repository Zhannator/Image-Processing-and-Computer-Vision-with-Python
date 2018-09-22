import numpy as np
from PIL import Image

# Open input images and convert them to image arrays
png_img_1 = Image.open('input/ps0-input-wide.png')
png_img_2 = Image.open('input/ps0-input-tall.png')
img_1_o = np.asarray(png_img_1)
img_2_o = np.asarray(png_img_2)

# 1A - Store two images as ps0-1-a-1.png and ps0-1-a-2.png in output folder
Image.fromarray(img_1_o).save('output/ps0-1-a-1.png')
Image.fromarray(img_2_o).save('output/ps0-1-a-2.png')

# 2A - Swap red and blue pixels of image 1
red, green, blue = png_img_1.split()
img_2_a_1 = Image.merge('RGB', (blue, green, red))
img_2_a_1.save('output/ps0-2-a-1.png')

# 2B - Create monochrome image (img1_green) by selecting green channel of image 1
img1_green = img_1_o.copy()
img1_green[:, :, 0] = 0 # Get rid of red
img1_green[:, :, 2] = 0 # Get rid of blue
Image.fromarray(img1_green).save('output/ps0-2-b-1.png')

# 2C - Create monochrome image (img1_red) by selecting red channel of image 1
img_2_c_1 = img_1_o.copy()
img_2_c_1[:, :, 1] = 0 # Get rid of green
img_2_c_1[:, :, 2] = 0 # Get rid of blue
Image.fromarray(img_2_c_1).save('output/ps0-2-c-1.png')

# 3A - Take the inner center square region of 100x100 pixels of monochrome version 
#      of image 1 and insert them into the center of monochrome version of image 2
# Confirm each images are at least 100x100 pixels
img_1_width, img_1_height = png_img_1.size
print "3A: img_1_width = " + str(img_1_width) + ", img_1_height = " + str(img_1_height)
img_2_width, img_2_height = png_img_2.size
print "3A: img_2_width = " + str(img_2_width) + ", img_2_height = " + str(img_2_height)
if ((img_1_width >= 100) and (img_1_height >= 100) and (img_2_width >= 100) and (img_2_height >= 100)):
    # Create img2_green
    img2_green = img_2_o.copy()
    img2_green[:, :, 0] = 0 # Get rid of red
    img2_green[:, :, 2] = 0 # Get rid of blue
    img2_green[(img_2_height/2 - 50):(img_2_height/2 + 50), (img_2_width/2 - 50):(img_2_width/2 + 50)] = img1_green[(img_1_height/2 - 50):(img_1_height/2 + 50), (img_1_width/2 - 50):(img_1_width/2 + 50)]
    Image.fromarray(img2_green).save('output/ps0-3-a-1.png')
else:
    print "Error: Image's width or height is too small for 3A."
    
# 4A - Min and max of pixel values of img1_green? Mean? Standard deviation?
img_1_pixel_min = np.amin(img1_green)
print "4A: img_1_pixel_min = " + str(img_1_pixel_min)
img_1_pixel_max = np.amax(img1_green)
print "4A: img_1_pixel_max = " + str(img_1_pixel_max)
img_1_mean = np.average(img1_green)
print "4A: img_1_mean = " + str(img_1_mean)
img_1_std = np.std(img1_green)
print "4A: img_1_std = " + str(img_1_std)

#4B - Subtract the mean from all pixels, then divide by standard deviation, then 
#     multiply by 10 (if your image is 0 to 255) or by 0.05 (if your image ranges 
#     from 0.0 to 1.0). Now add the mean back in.
img_4_b_1 = img1_green - img_1_mean
img_4_b_1 = img_4_b_1/img_1_std
print "4B: range [" + str(np.amin(img_4_b_1)) + " : " + str(np.amax(img_4_b_1)) + "]"
if (np.amax(img_4_b_1) > 1):
    img_4_b_1 = img_4_b_1*10
else:
    img_4_b_1 = img_4_b_1*0.05
img_4_b_1 = img_4_b_1 + img_1_mean
img_4_b_1 = np.round(img_4_b_1, decimals = 0) # Round to the nearest whole number
img_4_b_1 = img_4_b_1.astype('uint8') # Convert back from float to uint8
Image.fromarray(img_4_b_1).save('output/ps0-4-b-1.png')

# 4C - Shift img1_green to the left by 2 pixels
img_4_c_1 = img1_green[0:img_1_height, 2:img_1_width]
img_4_c_1 = np.hstack((img_4_c_1, img1_green[0:img_1_height, 0:2]))
Image.fromarray(img_4_c_1).save('output/ps0-4-c-1.png')

# 4D - Subtract shifted version of img1_green from the original and save the difference image
img_4_d_1 = img1_green - img_4_c_1
Image.fromarray(img_4_d_1).save('output/ps0-4-d-1.png')

# 5A - Take original colored image 1 and start adding Gaussian noise to the pixels in the green channel.
#      Increase sigma until noise is somewhat visible.
mu = 0
sigma = 0.5
noise = np.random.normal(mu, sigma, img_1_height*img_1_width)
img_5_a_1 = img_1_o.copy()
img_5_a_1[:, :, 1] = noise.reshape(img_1_height, img_1_width) + img_5_a_1[:, :, 1] # Add noise to green channel
Image.fromarray(img_5_a_1).save('output/ps0-5-a-1.png')

# 5B - Instead add that amount of noise to the blue channel
img_5_b_1 = img_1_o.copy()
img_5_b_1[:, :, 2] = noise.reshape(img_1_height, img_1_width) + img_5_b_1[:, :, 2] # Add noise to blue channel
Image.fromarray(img_5_b_1).save('output/ps0-5-b-1.png')