# Importing required modules
import cv2
import numpy as np
import math
import imutils
import os
import re
import pandas as pd


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(
      image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(rotated, (x, y), 100, (255, 0, 0), -1)
        mouseX, mouseY = x, y
        print("mouseX,mouseY", mouseX, mouseY)


def getRulerSmallMarkings(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    minThresh = 150
    ret, imgThresh = cv2.threshold(imgGray, minThresh, 255, 0)

    opening_kern_odd = 19
    kernel_opening = np.ones((opening_kern_odd, opening_kern_odd), np.uint8)
    imgOpen = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel_opening)

    closing_kern_odd = 19
    kernel_closing = np.ones((closing_kern_odd, closing_kern_odd), np.uint8)
    imgClose = cv2.morphologyEx(imgOpen, cv2.MORPH_CLOSE, kernel_closing)

    return imgClose


def caliberateScale(path):

    print("path for caliberation is : ", path)
    img = cv2.imread(path)


    img = cv2.resize(img, None, fx=0.1, fy=0.1)

    imgCopy = img.copy()
    imgClose = getRulerSmallMarkings(img)
  
    contours, heirarchy = cv2.findContours(
        imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle = 0
    cntNo = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(imgCopy, [box], 0, (0, 0, 255), 2)

        angle = rect[2]

        cv2.putText(imgCopy, str(cntNo), (int(rect[0][0]), int(
            rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cntNo += 1

    rotated = rotate_image(img, angle)


    centrCnt = getRulerSmallMarkings(rotated)

    cv2.imwrite("rotated.png", rotated)

    contourNew, heirarchy = cv2.findContours(
        centrCnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maxCnt = max(contourNew, key=cv2.contourArea)

    list_x_crop = []
    list_y_crop = []

    for cnt in contourNew:
        area = cv2.contourArea(cnt)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = rect[2]
        list_x_crop.append(box[0][0])
        list_x_crop.append(box[1][0])
        list_y_crop.append(box[1][1])
        list_y_crop.append(box[2][1])

        cntNo += 1

    list_x_crop.sort()
    list_y_crop.sort()

    imgCrop = rotated[:, list_x_crop[1]:list_x_crop[2]]
    crop_gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

    minThresh = 157
    ret, imgProcessedCrop = cv2.threshold(crop_gray, minThresh, 255, 0)


    opening_kern_odd = 4
    kernel_opening = np.ones((opening_kern_odd, opening_kern_odd), np.uint8)
    imgProcessedCrop = cv2.morphologyEx(
        imgProcessedCrop, cv2.MORPH_OPEN, kernel_opening)

    contours_crop, heirarchy = cv2.findContours(
        centrCnt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours_crop:
        area = cv2.contourArea(cnt)
        if area > 300:

            imgCrop = cv2.drawContours(
                imgCrop, cnt, -1, (255, 0, 0), 2, maxLevel=2)

    rotated_imageToMeasure = rotate_image(imgProcessedCrop, 90)


    hR, wR = rotated_imageToMeasure.shape


    ToAddtoStartBlack = True
    ToAddtoEndBlack = False
    startBlackList = []
    endBlackList = []
    cent_h = int(hR/2)
    for i in range(wR):

        if rotated_imageToMeasure[cent_h:cent_h+1, i] == 0 and ToAddtoStartBlack == True:
            startBlackList.append(i)
            ToAddtoStartBlack = False
            ToAddtoEndBlack = True
        if rotated_imageToMeasure[cent_h:cent_h+1, i] == 255 and ToAddtoEndBlack == True:
            endBlackList.append(i)
            ToAddtoStartBlack = True
            ToAddtoEndBlack = False

    print("*"*50)



    IndexDimNotToconsider = -1
    for i in range(len(endBlackList)):
        engravingWidth = endBlackList[i]-startBlackList[i]
        if engravingWidth > 10:
            IndexDimNotToconsider = i
            break
    listdistBetEngraving = []
    if IndexDimNotToconsider == -1:
        for i in range(len(endBlackList)):
            if i == IndexDimNotToconsider:
                continue
            else:
                distBetEngraving = startBlackList[i]-startBlackList[i-1]
                if distBetEngraving >= 0:
                    listdistBetEngraving.append(distBetEngraving)


    FinalDistBetweenEngraving = max(
        listdistBetEngraving, key=listdistBetEngraving.count)

    pixel_per_unit = FinalDistBetweenEngraving * 10

    print(f"pixel_per_unit is {pixel_per_unit}")

    return pixel_per_unit


def measure_l_b_h(path, pixel_per_unit):

    print("path for lbh measurement is : ", path)


    path2 = path
    image_l_W = cv2.imread(path2)
    image_l_W = cv2.resize(image_l_W, None, fx=0.1, fy=0.1)
    result_img = image_l_W.copy()
    h_n, w_n, ch = image_l_W.shape
    image_l_W[h_n-10:h_n, :w_n-1, :] = 0

    blur = cv2.blur(image_l_W, (3, 3))
    gray_im = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


    ret, threshold_image_l_W = cv2.threshold(gray_im, 54, 255, 0)

    im_floodfill = threshold_image_l_W.copy()
    h, w = threshold_image_l_W.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)


    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    angle = 0
    contour_img, heirarchy = cv2.findContours(
        im_floodfill_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxAreaCnt = max(contour_img, key=cv2.contourArea)
    rect = cv2.minAreaRect(maxAreaCnt)
    angle = rect[2]
    imgCp = image_l_W.copy()


    rotated_im = imutils.rotate_bound(im_floodfill_inv, 90-angle)
    box = []
    contour_img_rotated, heirarchy = cv2.findContours(
        rotated_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour_img_rotated:
        area = cv2.contourArea(cnt)
        if area > 100:
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            imgCp = cv2.drawContours(
                image_l_W.copy(), [box], 0, (0, 0, 255), 2)


    if len(box) > 3:
        list_x_box = []
        list_y_box = []
        for pnt in box:
            list_x_box.append(pnt[0])
            list_y_box.append(pnt[1])

        x_min = min(list_x_box)
        x_max = max(list_x_box)
        y_min = min(list_y_box)
        y_max = max(list_y_box)

    else:
        print("Contour not found... something went wrong ")

    left_img_cr = rotated_im[y_min:y_max, x_min:x_min+20]
    right_img_cr = rotated_im[y_min:y_max, x_max-20:x_max]
    top_img_cr = rotated_im[y_min:y_min+20, x_min:x_max]
    bottom_img_cr = rotated_im[y_max-20:y_max, x_min:x_max]
    list_imgs = [left_img_cr, right_img_cr, top_img_cr, bottom_img_cr]

    width_min_max = []
    length_min_max = []
    for i, img in enumerate(list_imgs):
        cntrs, heirarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxCnt = max(cntrs, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(maxCnt)
        if i == 0 or i == 1:
            width_min_max.append(h)
        elif i == 2 or i == 3:
            length_min_max.append(w)

    width_min_max.sort()
    length_min_max.sort()

    w_min_mm = width_min_max[0]/pixel_per_unit
    w_max_mm = width_min_max[1]/pixel_per_unit

    average_w_pixel = (width_min_max[0] + width_min_max[1]) / 2
    average_w_mm = round(average_w_pixel / pixel_per_unit, 2)

    l_min_mm = length_min_max[0]/pixel_per_unit
    l_max_mm = length_min_max[1]/pixel_per_unit

    average_l_pixel = (length_min_max[0] + length_min_max[1])/2
    average_l_mm = round(average_l_pixel / pixel_per_unit, 2)

    # print(f"width_min_max {width_min_max}")
    text1 = f"average width in pixel is {average_w_pixel} and in mm {average_w_mm}"
    # print(f"average width in pixel is {average_w_pixel} and in mm {average_w_mm}")
    text2 = f"average length in pixel is {average_l_pixel} and in mm {average_l_mm}"
    if average_l_mm < average_w_mm:
        print("swapping dimenstion L and W........................")
        a, b, c = w_min_mm, w_max_mm, average_w_mm

        w_min_mm, w_max_mm, average_w_mm = l_min_mm, l_max_mm, average_l_mm
        l_min_mm, l_max_mm, average_l_mm = a, b, c

    result_img = cv2.putText(result_img, text1, (10, 10),
                             cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)
    result_img = cv2.putText(result_img, text2, (10, 40),
                             cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)


    cv2.imshow("result", result_img)
    # # cv2.imshow("rotated",rotated)
    cv2.waitKey(1) 

    return average_w_mm, average_l_mm, w_min_mm, w_max_mm, l_min_mm, l_max_mm


def main():
    filename = "data.csv"

    # opening the file with w+ mode truncates the file
    f = open(filename, "w+")
    f.close()
    path = "./Crystals/"
    sub_folders = os.listdir(path)
    # print("sub folders :",sub_folders)

    data_line = ""
    with open("data.csv", "a+") as f:
        f.writelines(
            "Image for Caliberation,pixel per mm,img_w_l,min_w,max_w,avg_w,min_l,max_l,avg_l,img_th_l,min_w,max_w,avg_w,min_l,max_l,avg_l,area,volume\n")
        f.close()



    for folders in sub_folders:
        # print("each subfolder",folders)
        if os.path.isdir(path+folders):
            sub_folders_2 = os.listdir(path+folders)
            # print("each sub_folders_2",sub_folders_2)

            for folder in sub_folders_2:
                path_new = path+folders+"/"+folder+"/"
                if os.path.isdir(path_new):
                    files = os.listdir(path_new)
                    # print("files",files)

                    FilestoProcess = []

                    for file in files:

                        fileSplit = os.path.splitext(os.path.basename(file))
                        fileName, fileExt = fileSplit[0], fileSplit[1]

                        # print(f"fileName {fileName} fileExt {fileExt}")
                        if '.DS_Store' in fileName or '.DS_Store' in fileExt:
                            pass
                        else:
                            FilestoProcess.append(file)

                    list_fileNumbers = []
                    for each_filenames in FilestoProcess:
                        fileSplit = os.path.splitext(
                            os.path.basename(each_filenames))
                        fileName, fileExt = fileSplit[0], fileSplit[1]
                        # Splitting text and number in string
                        temp = re.compile("([a-zA-Z]+)([0-9]+)")
                        res = temp.match(fileName).groups()
                        list_fileNumbers.append(res[1])
                        # print("res",res)

                    # print("list_fileNumbers",list_fileNumbers)

                    zipped = zip(list_fileNumbers, FilestoProcess)

                    z = [x for _, x in sorted(zipped)]

                    print("sorted list is ", z)
                    FilestoProcess = z

                    ## start processing here
                    data_line = ""
                    pixel_per_unit = 155
                    average_w_mm, average_l_mm, average_t_mm = 0, 0, 0
                    for i, img_file in enumerate(FilestoProcess):
                        filePath = path_new+img_file
                        print("path", filePath)
                        data_line += img_file+","

                        if i == 0:

                            try:
                                pixel_per_unit = caliberateScale(filePath)
                                data_line += str(round(pixel_per_unit, 2)) + ","
                            except:
                                data_line += str(pixel_per_unit) + ","
                                print("something went wrong in caliberation")

                            print("Caliberation scale ended.....")
                            print("#"*25)

                        elif i == 1:

                            print(
                                "Width and length measurement started for second file")
                            average_w_mm, average_l_mm, w_min_mm, w_max_mm, l_min_mm, l_max_mm = measure_l_b_h(
                                filePath, pixel_per_unit)
                            print(
                                f"result is : average_w_mm {average_w_mm} and average_l_mm {average_l_mm}")
                            data_line += f"{w_min_mm}, {w_max_mm},{average_w_mm},{l_min_mm}, {l_max_mm},{average_l_mm},"

                        elif i == 2:
                            print(
                                "Width and length measurement started for second file")
                            average_t_mm, average_l_mm, t_min_mm, t_max_mm, l_min_mm, l_max_mm = measure_l_b_h(
                                filePath, pixel_per_unit)
                            print(
                                f"result is : average_t_mm {average_t_mm} and average_l_mm {average_l_mm}")
                            data_line += f"{t_min_mm}, {t_max_mm},{average_t_mm},{l_min_mm}, {l_max_mm},{average_l_mm},"

                            area = round(average_w_mm * average_l_mm, 2)
                            volume = round(
                                average_w_mm * average_l_mm * average_t_mm, 2)
                            data_line += f"{area},{volume}\n"
                            with open("data.csv", "a+") as f:
                                f.writelines(data_line)
                                f.close()
    print("-"*50)
    print("-"*50)
    file = "data.csv"
    df = pd.read_csv(file, error_bad_lines=False)
    pd.options.display.max_columns = len(df.columns)
    print(df)


if __name__ == '__main__':
    main()
