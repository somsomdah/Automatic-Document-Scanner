#Open Software Project - Final project / 홍유진 장다솜

import cv2
import numpy as np
import os
import argparse
from utils import find_vertices, resize_img



for file in os.listdir("./input"):
        filename,extention=file.split(".")

        # Read the image and resize it
        image=cv2.imread("./input/"+file)
        image=resize_img(image)

        # Show the resized original image.
        # cv2.imshow('INPUT',image)

        # Convert to grayscale and find edges
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        edge=cv2.Canny(blur,50,150)
        #cv2.imshow('Canny',edge)

        #Find and draw contours
        contours,_=cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,[0,255,0],2)
        #cv2.imshow('Contours',image)

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
        size=image.shape
        w,h,arr=find_vertices(approx)
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
        #cv2.imshow('OUTPUT',output)

        #Save the final output image (the scanned document) and finish
        cv2.imwrite('./output/out_'+file,output)
                

        cv2.waitKey(0)
        cv2.destroyAllWindows()
