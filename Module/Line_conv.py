import cv2
import numpy as np
from Module.find_skeleton import find_skeleton3

def hsl_average(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(32,32))  # 自适应直方图均衡化
    img_cll = clahe.apply(img_gray)

    return img_cll


def dev(img, kernel = 900): 

  len = np.shape(img)[0]

  dev_r = np.zeros(len) 
  
  for i in range(len-3):
    dev_r[i] = np.sum(img[i:i+3, 0:kernel]<117)

  return dev_r


def dev_col(img, kernel = 100): 

  len = np.shape(img)[1]

  dev_r = np.zeros(len) 
  
  for i in range(len-3):
    dev_r[i] = np.sum(img[0:kernel, i:i+3,]<117)

  return dev_r


def squeeze_num(dev, threshold = 0.9):

  col_num = np.squeeze(np.array(np.where(dev>(dev.max()*threshold))))
  
  return col_num

def self_cluster(arr):
  
  cluster = np.zeros([1], dtype=int)
  cluster = np.append(cluster, arr[0])
  #print(cluster[-1])

  for num, i in enumerate(arr):
    cal = abs(cluster[-1] - i)
    if cal > 5:
      cluster = np.append(cluster, i)
    else:
      cluster[-1] = i

  return cluster

def rotation_pixels(start, end):
  lens_start = np.shape(start)[0]
  lens_end = np.shape(end)[0]
  if lens_start > lens_end:
    aa = int((lens_end-1)/2)
    rotate = abs(start - end[aa])
  else:
    aa = int((lens_start-1)/2)
    rotate = abs(end - start[aa])
  
  return rotate.min()

def vertical_bias(img_cll):
    sample_start = np.array(img_cll[0:100, 0:40], dtype=int)
    dev_r2 = dev_col(sample_start)
    col_num_start = squeeze_num(dev_r2)
    cluster_start = self_cluster(col_num_start)

    sample_end = np.array(img_cll[400:500, 0:40], dtype=int)
    dev_r2 = dev_col(sample_end)
    col_num_end = squeeze_num(dev_r2)
    cluster_end = self_cluster(col_num_end)

    ro = rotation_pixels(cluster_start, cluster_end)

    return ro

def weft_bias(img_cll, line_num_true, ro_col, numCount):
    line_start = np.array(img_cll[:,0:1000], dtype=int)
    dev_temp = dev(line_start)
    line_start_num = squeeze_num(dev_temp)
    line_start_cluster = self_cluster(line_start_num)
    print("line_start_cluster:", line_start_cluster)

    line_midleft = np.array(img_cll[:,1000:2000], dtype=int)
    dev_temp = dev(line_midleft)
    line_midleft_num = squeeze_num(dev_temp)
    line_midleft_cluster = self_cluster(line_midleft_num)
    print('line_midleft_cluster:', line_midleft_cluster)

    line_midright = np.array(img_cll[:,2000:3000], dtype=int)
    dev_temp = dev(line_midright)
    line_midright_num = squeeze_num(dev_temp)
    line_midright_cluster = self_cluster(line_midright_num)
    print("line_midright_cluster:", line_midright_cluster)

    line_end = np.array(img_cll[:,3000:4000], dtype=int)
    dev_temp = dev(line_end)
    line_end_num = squeeze_num(dev_temp)
    line_end_cluster = self_cluster(line_end_num)
    print('line_end_cluster:', line_end_cluster)

    line_num = np.array([np.shape(line_start_cluster)[0], np.shape(line_midleft_cluster)[0], np.shape(line_midright_cluster)[0], np.shape(line_end_cluster)[0]])
    print('line_num', line_num)
    if (line_num.min() == 1) or (line_num.max() > line_num_true+1):
        print('misalignmen detected')
        cv2.imwrite("saved_alignment-" + str(numCount) +".png", img_cll)
    else:
    
        ro1 = rotation_pixels(line_start_cluster, line_midleft_cluster)
        ro2 = rotation_pixels(line_midleft_cluster, line_midright_cluster)
        ro3 = rotation_pixels(line_midright_cluster, line_end_cluster)

        ro = np.array([ro1, ro2, ro3], dtype=int)
        print('ro: ', ro)

        if ro.max()> (30 + ro_col):
            print('misalignmen detected')
            cv2.imwrite("saved_alignment-" + str(numCount) +".png", img_cll)
        else:
            print('Qulified')
            cv2.imwrite("saved_qualifiy-" + str(numCount) +".png", img_cll)

def weft_point(skeleton):
  
  aa = np.sum(skeleton, axis = 1)
  maxx = np.max(aa)

  weft_line = np.squeeze(np.where(aa > 0.8*maxx))

  #cluster = self_cluster(weft_line)
  #print(line_start_cluster)
  #cluster = cluster[1:]

  return weft_line



def misalignment_detector(img_gray, interval):
  
  for n, i in enumerate(range(300 , 500, 10)):
    print('round:', n)
    
    img_gray = np.asarray(img_gray)
    line_start = img_gray[10:80,i:i+interval]

    img_blur = cv2.blur(line_start, (20,2))

    #clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(32,32))  # 自适应直方图均衡化
    #img_cll = clahe.apply(img_blur)
    img_cll=cv2.normalize(img_blur,dst=None,alpha=400,beta=10,norm_type=cv2.NORM_MINMAX)
    #cv2_imshow(img_cll)

    _, thred = cv2.threshold(img_cll, 100, 255, 4)

    img_thred = cv2.blur(thred, (20,2))
    img_thred = cv2.blur(img_thred, (20,1))

    binary = cv2.adaptiveThreshold(np.uint8(cv2.bitwise_not(img_thred)),255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 77, 4)

    skeleton,iters = find_skeleton3(binary)

    #skeleton = 255 - skeleton

    if n == 0:                #初始化
      line_cluster = weft_point(skeleton)
      alignment = 0
      print('initial line:', line_cluster)

    else:
      if alignment:
        print('misalignment')
        break
      
      else:
        #print('current line:', line_cluster)
        line_cluster_temp = weft_point(skeleton)
        
        size_curent = np.size(line_cluster)   
        size_temp = np.size(line_cluster_temp)
        
        if (size_curent >= size_temp):      
          dev = 0
          if size_temp <= 1:
            aa_temp = np.min(abs(line_cluster - line_cluster_temp))
            if (aa_temp > 140):
              alignment = 1
          
          else: 
            for i in line_cluster_temp:
              aa_temp = np.min(abs(line_cluster - i))
              print('dev:', aa_temp)
              if (aa_temp > 140):
                alignment = 1
                break

        else: 
          temp = line_cluster 
          line_cluster = line_cluster_temp
          line_cluster_temp = temp

          #print('line change:', line_cluster)

          size_curent = np.size(line_cluster)   
          size_temp = np.size(line_cluster_temp)

          if size_temp <= 1:
            aa_temp = np.min(abs(line_cluster - line_cluster_temp))
            if (aa_temp > 140):
              alignment = 1
          
          else: 
            #print('line_cluster_temp', line_cluster_temp)
            for i in line_cluster_temp:
              aa_temp = np.min(abs(line_cluster - i))
              print('dev:', aa_temp)
              if (aa_temp > 140):
                alignment = 1
                break
      
      #print('qualify(0)/misalignment(1)', alignment)
  return alignment