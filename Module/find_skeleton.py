from timeit import default_timer as timer
import numpy as np
import cv2

def textile_enhance(img, kernel):
  img = img

  kernel = kernel
  #get the dimensions of the image
  n,m = np.shape(img)

  #initialize the edges image
  enhanced_edge_img = img.copy()

  #loop over all pixels in the image
  for row in range(3, n-2):
      for col in range(3, m-2):
          
          #create little local 3x3 box
          local_pixels = img[row-1:row+2, col-1:col+2]
          
          #apply the vertical filter
          transformed_pixels = kernel*local_pixels
          #remap the vertical score
          transformed_score = transformed_pixels.sum()/4
          enhanced_edge_img[row, col] = transformed_score
                   
  #remap the values in the 0-1 range in case they went out of bounds
  enhanced_edge_img = 255*enhanced_edge_img/enhanced_edge_img.max()

  return enhanced_edge_img

def cross_point(vertical_textile, horizontal_textile):
  vertical_textile = vertical_textile
  horizontal_textile = horizontal_textile

  #get the dimensions of the image
  n,m = np.shape(horizontal_textile)

  #initialize the edges image  
  edge_score = (vertical_textile**2 + horizontal_textile**2)**.5

  #remap the values in the 0-1 range in case they went out of bounds
  edge_score = 255*edge_score/edge_score.max()

  return edge_score

def edge_mask(img, kernel_window=(5,1), iter_num=2):

  new_image = cv2.convertScaleAbs(img, alpha=4, beta=10)

  # Load image, grayscale, Gaussian blur, Otsu's threshold

  blur = cv2.GaussianBlur(new_image, (3,3), 0)
  thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Detect horizontal lines
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_window)
  mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=iter_num)

  return mask


def edge_smooth(img, kernel=(100,5), thred_value=235, iters=1):
  
  img_smooth = img
  
  for i in range(iters):
    blur = cv2.blur(img_smooth, kernel)
    _, thred = cv2.threshold(blur, thred_value, 255, cv2.THRESH_BINARY_INV)
    
    img_smooth = thred

  return img_smooth


def time_fn(fn, img, iters=1):
    start = timer()
    result = None
    for i in range(iters):
        result = fn(img)
    end = timer()
    return (result,((end - start) / iters) * 1000)

def run_test(fn, img, i):
    res, t = time_fn(fn, img, 4)

def find_skeleton3(img):
    skeleton = np.zeros(img.shape,np.uint8)
    eroded = np.zeros(img.shape,np.uint8)
    temp = np.zeros(img.shape,np.uint8)

    _,thresh = cv2.threshold(img,127,255,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    iters = 0
    while(True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        #print(thresh.dtype)
        #print(temp.dtype)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton,iters)

def image_line_detect(img):
   img = np.asarray(img)
   img_blur = cv2.blur(img, (3,3))
   binary = cv2.adaptiveThreshold(np.uint8(cv2.bitwise_not(img_blur)),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, 0)
   img_blur1 = cv2.blur(binary, (33,5))
   #new_image = cv2.equalizeHist(img_blur)
   img_blur1 = cv2.blur(img_blur1, (33,1))
   img_blur1 = cv2.blur(img_blur1, (173,1))
   binary = cv2.adaptiveThreshold(np.uint8(cv2.bitwise_not(img_blur1)),255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,93, 0)
   img_blur2 = cv2.blur(binary, (33,5))
   #new_image = cv2.equalizeHist(img_blur)
   img_blur2 = cv2.blur(img_blur2, (63,1))
   img_blur2 = cv2.blur(img_blur2, (93,1))
   thresh1 = cv2.threshold(img_blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
   skeleton,iters = find_skeleton3(thresh1)

   return skeleton