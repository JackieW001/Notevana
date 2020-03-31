# import the necessary packages

import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import scipy.cluster.vq

def sort(a):
    n = len(a)
    #go through all the elements in the array
    for i in range(len(a)):
        #asuming that all the previous elements in the array are sorted
        #goes through 0 to one less than n
        for j in range(0, n-i-1):
            if a[j] > a[j+1]:
                temp = a[j]
                a[j] = a[j+1]
                a[j+1] = temp
    return a
# read in image to find contours on
image = cv2.imread('/Users/ishachirimar/noteshrink/page0000.png')

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0) #blur image to remove high frequency noise

#edged = cv2.Canny(gray, 75, 200) #edge detection
ret,thresh = cv2.threshold(gray,127,255,0)
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('closed', closing)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#find contours
_,contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours = sorted(contours, key = cv2.contourArea, reverse = True)
#cv2.imshow('contours', closing)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#arrays which will hold left most x and baseline y
xs = []
ys = []
centerx = []
centery = []

#dictionarly with bounding box corresponding to images
letters_by_bbox = {}
for i in range(0, len(contours)):
    cnt = contours[i]
    #mask = np.zeros(im2.shape,np.uint8)
    #cv2.drawContours(mask,[cnt],0,255,-1)
    
    x,y,w,h = cv2.boundingRect(cnt)
    bbox = cv2.boundingRect(cnt)
    #print(x,y,w,h)
    #add to arrays
    M = cv2.moments(cnt)
    if(i!= 0):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centerx.append(cX)
        centery.append(cY)
       
   
    xs.append(x)
    ys.append(y+h)
    #cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),3)
    #print(x+w, y+h)
   
    letter = thresh[y:y+h,x:x+w]
    if i != 0:
        letters_by_bbox[bbox] = letter
    
    
    
    #cv2.imshow('letter', letter)
    #cv2.waitKey(0)
    cv2.imwrite("numb" + str(i)+'.png', letter)
xs = np.array(xs, dtype=np.float)
ys = np.array(ys, dtype=np.float)
centerx = np.array(centerx, dtype=np.float)
centery = np.array(centery, dtype=np.float)

n_lines = 2
lines, _ = scipy.cluster.vq.kmeans(centery, n_lines, iter=1000)
lines = np.array(sorted(lines))

print('approx baseline of lines: ', repr(lines.astype(np.int16)))
plt.scatter(centerx, centery, s=50, c=np.argmin(np.abs(centery - lines[:, np.newaxis]), axis = 0), cmap = 'Paired')
plt.gca().invert_yaxis()
plt.autoscale(tight=True)
plt.hlines(lines, xmin = 0, xmax=centerx.max())
plt.show()
letters_by_line = [[] for _ in range(n_lines)]

for bbox, letter in list(letters_by_bbox.items()):
    nearest_line = np.argmin(np.abs([bbox[1] - lines]))
    letters_by_line[nearest_line].append([bbox, letter])

for line in letters_by_line:
    line.sort(key=lambda x:x[0][0])
    print('hi',line)
i = 0
for line in letters_by_line:
    for letter in line:
        cv2.imwrite('letter' + str(i) + '.png', letter[1])
        i += 1


'''
i = 0
for line in letters_by_line:
    for letter in line:
        #if i != (len(line) - 1):
            #print(letter[0][0], line[i+1][0][0])
        print(letter[0][0]) 
        np.sort(letter[0][0])
 #       i+=1
'''  

'''             
i = 0
for key in sorted(letters_by_bbox.iterkeys()):
    print key, letters_by_bbox[key]
    cv2.imwrite('letter' + str(i) + '.png', letters_by_bbox)
    i +=1
                               
'''                               
#print(letters_by_line)
        
#for i in range (len(letter_line)):
 #   print('1', letter_line[i][0][0])         


#cv2.imshow('Features', thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()