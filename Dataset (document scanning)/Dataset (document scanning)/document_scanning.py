#Open Software Project - Final project / 홍유진 장다솜

import cv2
import numpy as np
import os

def findVertices(pos):
	pts=[]
	n=len(pos) 
	
	for i in range(n):
		pts.append(list(pos[i][0]))
		# list of contour points {(x,y)}
		
	sums={}
	diffs={}
	
	for i in pts: # for every contour points
		x=i[0] # x value (column)
		y=i[1] # y value (row)
		sum=x+y 
		diff=y-x 
		sums[sum]=i 
		diffs[diff]=i 

	sums=sorted(sums.items())
	diffs=sorted(diffs.items())
	n=len(sums)
	
	rect=[sums[0][1], diffs[0][1], diffs[n-1][1], sums[n-1][1]]
	#       top-left        top-right       bottom-left     bottom-right
	
	h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2) #height of left side
	h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2) #height of right side
	h=max(h1,h2)
	
	w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)#width of upper side
	w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)#width of lower side
	w=max(w1,w2)
	
	return int(w),int(h),rect


# Read the image
img=cv2.imread('pic05.jpg')

# Image resizing if needed
if(img.shape[1]>1000 or img.shape[0]>1000):
        if (img.shape[1]>1000):
                r=1000.0 / img.shape[1]
                dim=(1000, int(img.shape[0] * r))
                img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)                
        elif (img.shape[0]>1000):
                r=1000.0 / img.shape[0]
                dim=(int(img.shape[1] * r),1000)
                img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)                

    
if(img.shape[1]<500 or img.shape[0]<500):
        if (img.shape[1]<500):
                r=1000.0 / img.shape[1]
                dim=(500, int(img.shape[0] * r))
                img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)                
        elif (img.shape[0]<500):
                r=500 / img.shape[0]
                dim=(int(img.shape[1] * r),500)
                img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                

# Show the resized original image.
cv2.imshow('INPUT',img)

# Convert to grayscale and find edges
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)
edge=cv2.Canny(blur,50,150)
cv2.imshow('Canny',edge)

#Find and draw contours
contours,_=cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,[0,255,0],2)
cv2.imshow('Contours',img)

#Find the part of the document(maximum area) in the image devided by contours
n=len(contours)
max_area=0
pos=0

for i in contours:
	area=cv2.contourArea(i)
	if area>max_area:
		max_area=area
		pos=i
		
		
# Find the vertices and the size of the object	
peri=cv2.arcLength(pos,True)
approx=cv2.approxPolyDP(pos,0.02*peri,True)
size=img.shape
w,h,arr=findVertices(approx)
# findVertices() :returns the vertices and the size(hight, width) of the object

# Find the perspective matrix
pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
pts1=np.float32(arr)
M=cv2.getPerspectiveTransform(pts1,pts2)
                
# Adjust the image into a rectangular form with perspective transformation
fitted=cv2.warpPerspective(gray,M,(w,h))
#cv2.imshow('FITTED',fitted)

#Rivision process
if(0):  # binary output
        # Make the document clear with adaptive thresholding using moving averages
        # and make letters smoother by Gaussian blurring
        output=cv2.adaptiveThreshold(fitted, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 7, 12)
        output=cv2.GaussianBlur(output,(3,3),0)

else : # grayscale output
        # using unsharp masking
        k=0.7
        blurred=cv2.GaussianBlur(fitted,(5,5),0)
        output=cv2.addWeighted(fitted, 1/(1-k), blurred, -k/(1-k), 0)

output = cv2.resize(output,(w,h),interpolation = cv2.INTER_AREA)

#Show the final output image (the scanned document)
cv2.imshow('OUTPUT',output)

#Save the final output image (the scanned document) and finish
cv2.imwrite('output.jpg',output)
        

cv2.waitKey(0)
cv2.destroyAllWindows()
