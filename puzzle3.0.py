# -*- coding:utf-8 -*-



import cv2

import numpy as np
img = cv2.imread('image3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
#SIFT
sift = cv2.SIFT()
# kp 是所有128特征描述子的集合
kp = sift.detect(gray, None)
#print len(kp)
 
# 找到后可以计算关键点的描述符
Kp,res = sift.compute(gray, kp)
#print Kp    # 特征点的描述符
#print res   # 是特征点个数*128维的矩阵

kp2,res1 = sift.detectAndCompute(gray,None)
#print "******************************"
#print res1
 
img = cv2.drawKeypoints(img, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = (51, 163, 236))

cv2.imwrite('sift.png', img)
'''
'''
#SURF
# surf.hessianThreshold=1000
surf = cv2.SURF(1000)
 
kp,res = surf.detectAndCompute(gray,None)

#print res.shape

img = cv2.drawKeypoints(img,kp,None,(255,0,255),4)

cv2.imwrite('SURF.png', img) 
'''

#Harris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 23, 0.04)
img[dst>0.01 * dst.max()] = [255, 0, 255] 
cv2.imwrite('Harris.png', img)




