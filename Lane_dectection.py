import numpy as np
import cv2
from imageai.Detection import ObjectDetection
import os
from PIL import ImageGrab

#fourcc = cv2.VideoWriter_fourcc('X','V','I','D') #you can use other codecs as well.
#vid = cv2.VideoWriter('record.avi', fourcc, 8, (500,490))
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
execution_path = os.getcwd()


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
import math
time.sleep(5)
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
 
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if len(lines)==1:
      return img   
    else:
     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
     draw_lines(line_img, lines)
     return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
import os
while(True):
    img1 = ImageGrab.grab(bbox=(0,200,800,400)) #x, y, w, h
    img_np = np.array(img1)
#    frame = img_np
    imshape = img_np.shape
    vertices = np.array([[(0,imshape[0]),(-200, 400), (1000, 400), (400, 100)]], dtype=np.int32)
    # convert the rgb image to grayscale
    gray = grayscale(img_np)
    # add gaussian blur to gray image
    gray_blur = gaussian_blur(gray, 3)
    # use cannay edge detection to gray_blur
    edges = canny(gray_blur, 50, 100)
    # get the masked image of edges
    masked_image = region_of_interest(edges, vertices)
    # use hough transform to draw lines
    hough_line_img = hough_lines(masked_image, 1, np.pi/180, 50, 10, 200)
    #draw lines on the original image
    combo = weighted_img(hough_line_img, img_np)
#    vid.write(img_np)
    cv2.imshow("frame", combo)
    key = cv2.waitKey(1)
    if key == 27:
        break    

#vid.release()
cv2.destroyAllWindows()