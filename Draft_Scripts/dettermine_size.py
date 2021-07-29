import cv2

img = cv2.imread("DDAB.png")
gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cny = cv2.Canny(gry, 50, 200)
lns = cv2.ximgproc.createFastLineDetector().detect(cny)
cpy = img.copy()

(height, width) = cpy.shape[:2]

mx = width
my = height
Mx = 0
My = 0

for ln in lns:
    a = int(ln[0][0])
    y1 = int(ln[0][1])
    b = int(ln[0][2])
    y2 = int(ln[0][3])

    mx = min(a, b, mx)
    my = min(y1, y2, my)
    Mx = max(a, b, Mx)
    My = max(y1, y2, My)

    print("Coords: ({}, {})->({}, {})".format(a, y1, b, y2))


cv2.line(cpy, pt1=(mx, my), pt2=(Mx, my), color=(0, 255, 0), thickness=10)
cv2.putText(cpy, '({}, {})'.format(mx, my), (mx - 10, (my - 30)), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 255, 0), 3, cv2.LINE_AA)
cv2.putText(cpy, '({}, {})'.format(Mx, my), (Mx - 340, (my - 30)), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 255, 0), 3, cv2.LINE_AA)
cv2.putText(cpy, '{} pixel'.format((Mx - mx)), (int((Mx + mx)/2), (my - 30)), cv2.FONT_HERSHEY_SIMPLEX,
            2, (0, 255, 0), 3, cv2.LINE_AA)
cv2.line(cpy, pt1=(mx, my), pt2=(mx, My), color=(0, 0, 255), thickness=10)
cv2.putText(cpy, '({}, {})'.format(mx, my), (mx - 10, (my + 50)), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 0, 255), 3, cv2.LINE_AA)
cv2.putText(cpy, '({}, {})'.format(mx, My), (mx - 10, (My - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 0, 255), 3, cv2.LINE_AA)
cv2.putText(cpy, '{} pixel'.format(My - my), (mx - 150, int((My + my)/2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 0, 255), 3, cv2.LINE_AA)
cv2.imshow("cpy", cpy)
cv2.waitKey(0)
