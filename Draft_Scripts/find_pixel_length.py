import cv2
import numpy as np


#smallest division size of the ruler
real_size_in_mm=20
pixel_distance=float(input("Input the distance of the ruler's division in pixels: "))


## 1. read image here
# img = cv2.imread("1.jfif")


img = cv2.imread("images/2.png")
# img = cv2.imread("3.png")

img = cv2.resize(img,(640,480))

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)

threshold1 = 70
threshold2 = 255
minArea = 500
maxArea = 640*480*0.9

def findDistance(pt1,pt2):
    dist = ((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)**(1/2)

    return dist

while True:
    ret, imgThreshold = cv2.threshold(imgBlur, threshold1,threshold2,0)    
    
    cv2.imshow("imgThreshold",imgThreshold)
    contours, heirarchy = cv2.findContours(imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("contours length", len(contours))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < maxArea and area > minArea:
            x,y,w,h = cv2.boundingRect(cnt)    
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)            
            box = np.int0(box)            
            rotAngle = rect[2]  
            wR,hR = rect[1]

            img = cv2.drawContours(img,[box],0,(0,0,255),2)
            print("box",box)
            
            for pts in box:
                img = cv2.putText(img, str(pts),(pts[0]-5,pts[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1)


            # for pt2 and pt2    
            dist1 = int(findDistance(box[1],box[0]))

            print ("dist1",dist1)
            print("dist1_in_mm", dist1*real_size_in_mm/pixel_distance)
            pt1 = (box[1][0]-5,box[1][1]-5)
            pt2 = (box[0][0]-5,box[0][1]-5)            
            center1 = (int((box[1][0]+box[0][0])/2),int((box[1][1]+box[0][1])/2))

            cv2.arrowedLine(img, pt1, pt2, (0,255,0), 1)
            cv2.arrowedLine(img, pt2, pt1, (0,255,0), 1)
            
            img = cv2.putText(img, str(dist1),(center1[0]-10,center1[1]),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)

            # for pt3 and pt2

            dist2 = int(findDistance(box[2],box[1]))

            print ("dist2",dist2)
            print("dist2_in_mm", dist2*real_size_in_mm/pixel_distance)
            pt3 = (box[2][0],box[2][1]-10)
            pt4 = (box[1][0],box[1][1]-10)
            center2 = (int((box[2][0]+box[1][0])/2),int((box[2][1]+box[1][1])/2))

            cv2.arrowedLine(img, pt3, pt4, (0,255,0), 1,tipLength=0.01)
            cv2.arrowedLine(img, pt4, pt3, (0,255,0), 1,tipLength=0.01)
            img = cv2.putText(img, str(dist2),(center2[0]+10,center2[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)


     
    cv2.imshow("img",img)

    key = cv2.waitKey(-1)
    if key == ord('q'):
        break


cv2.destroyAllWindows()
