import numpy as np
import cv2  
import matplotlib.pylab as plt
img=plt.imread('D:/04.jpg')
#cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)  
#cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1)  
#cv2.ellipse(img,(256,256),(100,50),0,0,0,255,-1)  
font=cv2.FONT_HERSHEY_SIMPLEX  
cv2.putText(img,"opencv",(2,25),font,1,(255,255,255),2)  
cv2.putText(img,"opencv",(2,45),font,1,(255,255,255),2)  
plt.imshow(img)  
plt.show()
