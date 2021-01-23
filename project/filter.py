
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from skimage import morphology
import skimage
#1
# # img=cv2.imread('DDAB.png')
# img=cv2.imread('DSCN2879.JPG')
# # img=cv2.imread('DSCN2881.JPG')
# # img=cv2.imread('DMNAB.png')
#
# canny=cv2.Canny(img,80,120)
# titles=['images', 'canny']
# images=[img, canny]
# for i in range(2):
#     plt.subplot(1,2,i+1), plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
#
# plt.show()



# cv2_img=cv2.imread('DSCN2879.JPG')
# cv2.imshow('Apple', cv2_img)
# cv2.waitKey()



# #2
# cv2_img_gray=cv2.imread('DDAB.png',0)
# ret, thresh=cv2.threshold(cv2_img_gray,70,150,cv2.THRESH_BINARY)
#
# titles=['Original image', 'Binary']
# images=[cv2_img_gray, thresh]
#
# for i in range(2):
#     plt.subplot(2,3, i+1), plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


#3
image = cv2.imread("DDAB.png")
# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# create a binary thresholded image
_, binary = cv2.threshold(gray, 127, 127, cv2.THRESH_BINARY_INV)
# show it
plt.imshow(binary, cmap="gray")
plt.show()
# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw all contours
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# show the image with the drawn contours
plt.imshow(image)
plt.show()


# #4
# img = cv2.imread('DDAB.png')
# img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 140).astype('uint8')
#
# se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
#
# mask = np.dstack([mask, mask, mask]) / 255
# out = img * mask
#
# cv2.imshow('Output', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('output.png', out)



# #5
# # Load image, convert to grayscale, Gaussian blur, Otsu's threshold
# image = cv2.imread('DDAB.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
# # Filter using contour area and remove small noise
# cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 5500:
#         cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
#
# # Morph close and invert image
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#
# cv2.imshow('thresh', thresh)
# cv2.imshow('close', close)
# cv2.waitKey()
