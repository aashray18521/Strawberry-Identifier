from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

green = (0, 255, 0)

def show(image):
    
    #get figure size in inches
    plt.figure(figsize=(10,10))
    plt.imshow(image, interpolation='nearest')
    
##applying mask to image
def overlay_mask(mask, image):
    #make the mask rgb
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    #calculating weighted sum of two arrays (image arrays)
    #adding mask over the image
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    #copy image 
    image = image.copy()
    
    #this function gives us all the contours
    #CHAIN_APPROX_SIMPLE returns only the end points
    #we are going to use retrive list (RETR_LIST) to get all the contour approximations
    #contour will get these and hierarchy will get us the chain of contours from greatest to least (ie, the size of these ellipses from greatest to least)
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #isolating the largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key = lambda x: x[0])[1]  #[1] represents contour with max value
    
    #return the biggest contour
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask
    
def circled_contour(image, contour):
    #this is where we define the shape of that contour
    #get the bounding ellipse
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    
    #take the image with ellipse on it make in 'green' with value of 2(width of contour line )
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.CV_AA)
    return image_with_ellipse
    

def find_strawberry(image):
    #RGB is Red Green Blue
    #BGR is Blue Green Red
    #its about ordering of colors
    #RGB color scheme is better coz Blue occupies least significant area
    ######-----------convert to correct color scheme------###############
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #########-------scale out image properly------###########
    #to find the size of image
    max_dimension = max(image.shape)
    #max window size that we are going to use is 700x660 pixels and we wanna make our strawberry image fit in this image
    scale = 700/max_dimension
    #making width and height same scaled , ie , square instead of rectangle
    image = cv2.resize(image, None, fx = scale, fy = scale)
    
    #######------Clean our image--------########
    #applying gaussian blur to smooth colors in our image & to remove noise
    image_blur = cv2.GaussianBlur(image, (7,7), 0) #(image, kernel size, how much we ant to filter it(optional))
    #convert color scheme again
    #to separate image intensity from color information
    #we want to focus only on the color that's why we are converting this
    #hue saturation valve
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    
    
    #########---------Define our filters-------#########
    #filter by color  (color is separate from brightness)
    #we want to detect strawberry in a certain color range (certain redness)
    #
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    
    #mask = filter (to focus on one color and blur everything else)
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)
    
    #filter by brightness
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)
    
    #take these two masks and combine our masks
    
    mask = mask1 +mask2
    
    #########--------Segmentation----------##############
    #we are going to use these mask to separate strawberry from everything else
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    
    #closing operation - dialation followed by erosion
    #useful for closing small holes inside foreground of objects
    #like small black points in object, it further helps refine that smoothness
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    #opening operation - erosion followed by dialation
    #helps in removing noise
    #both closing and open operations add to each other, ie, they complement each other
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    
    #########--------Find the biggest strawberry----------###############
    #find the biggest ellipse for the strawberry and return the mask for that strawberry
    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)
    
    
    ########----------overlay the masks that we created on the image-----------##################
    overlay = overlay_mask(mask_clean, image)
    
    
    ########--------circle the biggest one---------##########
    circled = circled_contour(overlay, big_strawberry_contour)
    
    show(circled)
    
    
    #########----------convert back to original color scheme--------########
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
    
    return bgr
    
    
    
    
    
    
#read the image
image = cv2.imread(“Your image path”)
result = find_strawberry(image)
#write it
cv2.imwrite(“Your image saving path”, result)