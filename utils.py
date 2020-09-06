import numpy as np
import cv2

def find_vertices(pos):
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

def resize_img(image):
     img=image
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
     return img
                
