#M16
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from skimage import morphology
import skimage  
import math


## 1. read image here
img = cv2.imread("DSCN2882.JPEG")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold1 = 87
threshold2 = 255
ret, imgThreshold = cv2.threshold(imgGray, threshold1, threshold2, 0)

img = cv2.resize(img, (0, 0), fx=0.20, fy=0.20)
imgCopy = img.copy()
imgThreshold = cv2.resize(imgThreshold, (0, 0), fx=0.20, fy=0.20)

### 1. To orient image so that engravings are perfectly vertical

h, w = imgThreshold.shape
print("h,w", h, w)
w1 = 0
w2 = 0
for i in range(w):
    # imgThreshold[h-12,i] =0
    if imgThreshold[h - 10, i] == 0:
        w1 = i
        print("w1", w1)

        break

for i in range(w):
    # imgThreshold[h-22,i] =0
    if imgThreshold[h - 20, i] == 0:
        w2 = i
        print("w2", w2)

        break

# inclination angle found here
theta = math.degrees(math.atan(10 / (w1 - w2)))
print("theta", theta)

rotated_image = rotate_image(imgThreshold, ((90 - theta) * -1))
rotated_Org_image = rotate_image(imgCopy, ((90 - theta) * -1))

cv2.namedWindow('rotated_Org_image')
cv2.setMouseCallback("rotated_Org_image", mouse_click)

#### 2. getting distance between the engravings

# 2.1 getting first white from bottom left most corner

hR, wR = rotated_image.shape
print("hR, wR", hR, wR)

ToAddtoStartBlack = True
ToAddtoEndBlack = False
startBlackList = []
endBlackList = []

for i in range(wR):
    # rotated_image[400:401,:wR-1]=0
    if rotated_image[400:401, i] == 0 and ToAddtoStartBlack == True:
        startBlackList.append(i)
        ToAddtoStartBlack = False
        ToAddtoEndBlack = True
    if rotated_image[400:401, i] == 255 and ToAddtoEndBlack == True:
        endBlackList.append(i)
        ToAddtoStartBlack = True
        ToAddtoEndBlack = False

print("startBlackList ",startBlackList)
print("endBlackList",endBlackList)


# 2.2 logic for checking unwanted entry of pixels in width check

IndexDimNotToconsider = -1
for i in range(7):
    engravingWidth = endBlackList[i] - startBlackList[i]
    if engravingWidth > 5:
        IndexDimNotToconsider = i
        break
listdistBetEngraving = []
if IndexDimNotToconsider == -1:
    for i in range(1, 7):
        distBetEngraving = startBlackList[i] - startBlackList[i - 1]
        listdistBetEngraving.append(distBetEngraving)

print("listdistBetEngraving", listdistBetEngraving)

# 2.3 Finally dist between each engraving is below
FinalDistBetweenEngraving = max(listdistBetEngraving, key=listdistBetEngraving.count)
print("listdistBetEngraving", FinalDistBetweenEngraving)

# 2.4 displaying arrow and dimension in pixels in image
pt1 = (startBlackList[0], 400)
pt2 = (startBlackList[1], 400)

rotated_Org_image = drawArrowedLines(rotated_Org_image, pt1, pt2, FinalDistBetweenEngraving)
rotated_Org_image = cv2.putText(rotated_Org_image, f"Resized image to : {w} x {h} pixels", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("Thresh", imgThreshold)
cv2.imshow("img", img)
# cv2.imshow("rotated_image",rotated_image)
cv2.imshow("rotated_Org_image", rotated_Org_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
