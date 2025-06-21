import cv2 as cv
import numpy as np
print("FPS SET TO 25 FOR ACCURACY")


width = 640
height = 360
X = 0
Y = 0

Lhue = 0
Hhue = 0
Lsat = 00
Hsat = 0
Lval = 0
Hval = 0

click = False

cap = cv.VideoCapture("warraaathoooose.mp4")


def T3(x):
    global Lsat
    Lsat = x
    print(Lsat)


def T4(x):
    global Hsat
    Hsat = x
    print(Hsat)


def T1(x):
    global Lhue
    Lhue = x
    print(Lhue)


def T2(x):
    global Hhue
    Hhue = x
    print(Hsat)


def T5(x):
    global Lval
    Lval = x
    print(Lval)


def T6(x):
    global Hval
    Hval = x
    print(Hval)


def CL(event, x, y, flags, param):
    global X, Y, click

    if event == cv.EVENT_LBUTTONDOWN:
        X = x
        Y = y
        print("Cord of point clicked : ")
        print(X, Y)
        click = True


cv.namedWindow("Video")
cv.namedWindow("trackbar")
cv.resizeWindow("trackbar", 400, 225)
cv.moveWindow("trackbar", width, 0)
cv.createTrackbar('Lhue', "trackbar", 0, 179, T1)
cv.createTrackbar('Hhue', "trackbar", 0, 179, T2)
cv.createTrackbar('Lsat', "trackbar", 0, 255, T3)
cv.createTrackbar('Hsat', "trackbar", 0, 255, T4)
cv.createTrackbar('Lval', "trackbar", 0, 255, T5)
cv.createTrackbar('Hval', "trackbar", 0, 255, T6)

cv.setMouseCallback("Video", CL)


while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        continue
    framee = cv.resize(frame, (width, height))
    framehsv = cv.cvtColor(framee, cv.COLOR_BGR2HSV)
    lowerb = np.array([Lhue, Lsat, Lval])
    highrb = np.array([Hhue, Hsat, Hval])
    mask = cv.inRange(framehsv, lowerb, highrb)
    smallmask = cv.resize(mask, (width // 2, height // 2))
    obj = cv.bitwise_and(framehsv, framehsv, mask=mask)
    smlallobj = cv.resize(obj, (width // 2, height // 2))

    key = cv.waitKey(40)

    if click == True:
        if 0 <= X <= width and 0 <= Y <= height:
            hsvclicked = framehsv[Y, X]
            print(hsvclicked)
            print("prenti mara bark nkmk")
            h = int(hsvclicked[0])
            s = int(hsvclicked[1])
            v = int(hsvclicked[2])
            Lhue = max(h - 10, 0)
            Hhue = min(h + 10, 255)
            Lsat = max(s - 50, 0)
            Hsat = min(s + 50, 255)
            Lval = max(v - 50, 0)
            Hval = min(v + 50, 255)
            cv.setTrackbarPos('Lhue', "trackbar", Lhue)
            cv.setTrackbarPos('Hhue', "trackbar", Hhue)
            cv.setTrackbarPos('Lsat', "trackbar", Lsat)
            cv.setTrackbarPos('Hsat', "trackbar", Hsat)
            cv.setTrackbarPos('Lval', "trackbar", Lval)
            cv.setTrackbarPos('Hval', "trackbar", Hval)
            click = False

    cnt, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if cnt:
        Lacnt = max(cnt, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(Lacnt)
        cx=x+w//2
        cv.rectangle(framee, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.rectangle(smallmask, (x // 2, y // 2), ((x + w) // 2, (y + h) // 2), (255), 2)
        cv.rectangle(smlallobj, (x // 2, y // 2), ((x + w) // 2, (y + h) // 2), (255), 2)
        CF=int(cap.get(cv.CAP_PROP_POS_FRAMES))
        TF=int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if CF==TF-1:
            if 0<=cx<=width//2:
                print("the correct cup is on the left side")
            elif width//2<=cx<=(2*width)//2:
                print("the correct cup is on the middle side")
            else:
                print("the correct cup is on the right side")





    cv.rectangle(framee, (0, 0), (640//3,360), (0, 0, 0), 2)
    cv.rectangle(framee, (640//3,0), (1280//3, 360), (0, 0, 0), 2)
    cv.rectangle(framee, (1280//3, 0), (640, 360), (0, 0, 0), 2)

    cv.imshow("Video", framee)
    cv.imshow("obj", smlallobj)
    cv.imshow("Mask", smallmask)
    cv.moveWindow("Video", 0, 0)
    cv.moveWindow("Mask", 0, height)
    cv.moveWindow("obj", width // 2, height)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()


# hue low	100
# hue high	130
# sat low	100
# sat high	255
# val low	50
# val high	255