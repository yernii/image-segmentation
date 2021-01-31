import cv2

img = cv2.imread("DDAB.png")
gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cny = cv2.Canny(gry, 50, 200)
lns = cv2.ximgproc.createFastLineDetector().detect(cny)
cpy = img.copy()

(h, w) = cpy.shape[:2]

min_x = w
min_y = h
max_x = 0
max_y = 0

for ln in lns:
    x1 = int(ln[0][0])
    y1 = int(ln[0][1])
    x2 = int(ln[0][2])
    y2 = int(ln[0][3])

    min_x = min(x1, x2, min_x)
    min_y = min(y1, y2, min_y)
    max_x = max(x1, x2, max_x)
    max_y = max(y1, y2, max_y)

    print("Coords: ({}, {})->({}, {})".format(x1, y1, x2, y2))


cv2.line(cpy, pt1=(min_x, min_y), pt2=(max_x, min_y), color=(0, 255, 0), thickness=10)
cv2.putText(cpy, '({}, {})'.format(min_x, min_y), (min_x - 10, (min_y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 255, 0), 3, cv2.LINE_AA)
cv2.putText(cpy, '({}, {})'.format(max_x, min_y), (max_x - 340, (min_y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 255, 0), 3, cv2.LINE_AA)
cv2.putText(cpy, '{} pixel'.format((max_x - min_x)), (int((max_x + min_x)/2), (min_y - 30)), cv2.FONT_HERSHEY_SIMPLEX,
            2, (0, 255, 0), 3, cv2.LINE_AA)
cv2.line(cpy, pt1=(min_x, min_y), pt2=(min_x, max_y), color=(0, 0, 255), thickness=10)
cv2.putText(cpy, '({}, {})'.format(min_x, min_y), (min_x - 10, (min_y + 50)), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 0, 255), 3, cv2.LINE_AA)
cv2.putText(cpy, '({}, {})'.format(min_x, max_y), (min_x - 10, (max_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 0, 255), 3, cv2.LINE_AA)
cv2.putText(cpy, '{} pixel'.format(max_y - min_y), (min_x - 150, int((max_y + min_y)/2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 0, 255), 3, cv2.LINE_AA)
cv2.imshow("cpy", cpy)
cv2.waitKey(0)
