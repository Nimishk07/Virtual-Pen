import cv2
import numpy as np
import time
def nothing(x):
    pass
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')
color_from_disk=True    # color shade card true that means that shade card is already present to choose color from else prepare a shade card
if color_from_disk:
    color=np.load('Shadecard.npy')
else:
    # A white canvas for picture card
    color = np.full((720, 1280, 3), 255, dtype=np.uint8)
    # To set the variations in the brightness levels
    for row in color:
        bright=0
        for pixel in row:
            pixel[1] = int(bright / 5)
            bright += 1
    hue = 0
    # To set the variations in the color levels
    for row in color:
        hue += 1
        for pixel in row:
            pixel[0] = int(hue/ 4)
    np.save('Shadecard', color)
select_color=False
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
inspect=0
kernel = np.ones((5, 5), np.uint8)
on =False
# Initializing the canvas on which we will draw upon
canvas = None

# Initilize x1,y1 points
x1, y1 = 0, 0
r,g,b=(255,0,0)
sr,sg,sb=r,g,b
# Threshold for noise
noiseth = 800
clear_area=100000
clear=False
current="pen"
working=True
cv2.namedWindow('virtual pen', cv2.WINDOW_NORMAL)
cv2.createTrackbar("val", 'virtual pen', 0, 255, nothing)
cv2.createTrackbar("thickness", 'virtual pen', 1, 200, nothing)
pen=cv2.resize(cv2.imread("pen.jpg"),(100,100))
eraser=cv2.resize(cv2.imread("eraser.jpg"),(100,100))
lower_limit = np.array([2,2,2])
upper_limit = np.array([255, 255, 255])
mask3=cv2.inRange(pen,lower_limit,upper_limit)
mask2=cv2.inRange(eraser,lower_limit,upper_limit)
apen=cv2.bitwise_and(pen,pen,mask=mask3)
aeraser=cv2.bitwise_and(eraser,eraser,mask=mask2)
while (1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Initialize the canvas as a black image of the same size as the frame.
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # If you're reading from memory then load the upper and lower ranges
    # from there
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]

    # Otherwise define your own custom values for upper and lower range.
    else:
        lower_range = np.array([26, 80, 147])
        upper_range = np.array([81, 255, 255])

    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Perform morphological operations to get rid of the noise
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make sure there is a contour present and also its size is bigger than
    # the noise threshold.
    if not select_color:
        if contours and cv2.contourArea(max(contours,key=cv2.contourArea)) > noiseth:

            c = max(contours, key=cv2.contourArea)
            x2, y2, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # If there were no previous points then save the detected x2,y2
            # coordinates as x1,y1.
            # This is true when we writing for the first time or when writing
            # again when the pen had disappeared from view.
            if x1 == 0 and y1 == 0:
                x1, y1 = x2, y2

            else:
                # Draw the line on the canvas and eraser
                if working:
                    if current == "pen":
                        thick=cv2.getTrackbarPos("thickness","virtual pen")
                        canvas = cv2.line(canvas, (x1, y1), (x2, y2), [sb, sg, sr], thick)
                        if area > clear_area:
                            cv2.putText(canvas, 'Clearing Canvas', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5,
                                        cv2.LINE_AA)
                            clear = True
                    else:
                        canvas = cv2.circle(canvas, (x1, y1), 20, (0, 0, 0), -1)

            # After the line is drawn the new points become the previous points.
            x1, y1 = x2, y2
            if area>clear_area:
                cv2.putText(canvas, 'Clearing Canvas', (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
                clear = True
        else:
            # If there were no contours detected then make x1,y1 = 0
            x1, y1 = 0, 0

        # Merge the canvas and the frame.
        frame = cv2.add(frame, canvas)
        frame = cv2.add(frame, canvas)
        if current != "pen":
            cv2.circle(frame, (x1, y1), 20, (255, 255, 255), -1)
    else:
        x3,y3=0,0
        if contours and cv2.contourArea(max(contours,key=cv2.contourArea)) > noiseth:
            c = max(contours, key=cv2.contourArea)
            x3, y3, w, h = cv2.boundingRect(c)
        value = cv2.getTrackbarPos("val", "virtual pen")
        color[5:715, 5:1275, 2] = value
        colo = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        frame=colo
        b,g,r=tuple(colo[y3][x3])
        b=int(b)
        g=int(g)
        r=int(r)
        cv2.circle(frame,(x3,y3),20,[b,g,r],-1)
        cv2.circle(frame,(x3,y3),22,[0,0,0],2)

    if not select_color:
        if working:
            if current=="pen":
                mask4 = cv2.bitwise_not(mask3)
                fra=frame[620:720, 0:100]
                f = cv2.bitwise_and(fra, fra, mask=mask4)
                fr=f+apen
                frame[620:720, 0:100]=fr
            else:
                mask4 = cv2.bitwise_not(mask2)
                fra=frame[620:720, 0:100]
                f = cv2.bitwise_and(fra, fra, mask=mask4)
                fr=f+aeraser
                frame[620:720, 0:100]=fr

    cv2.imshow('virtual pen', frame)
    stacked = np.hstack((canvas, frame))
    if inspect:
        cv2.imshow('inspect', cv2.resize(stacked, None, fx=0.6, fy=0.6))
        on=True
    elif not inspect and on:
        cv2.destroyWindow('inspect')
        on=False


    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if clear==True:
        time.sleep(1)
        canvas = None
        clear = False
    # when b is pressed the mode will change to a color selection mode
    if k==ord('b'):
        if select_color:
            select_color=False
            working=False
        else:
            select_color=True
    #when you hit spacebar then it will change to eraser mode
    if k== ord(' '):
        if current!="pen":
            current="pen"
        else:
            current="eraser"
    if k==ord('a'):
        if working:
            working=False
        else:
            working=True
    # When c is pressed clear the canvas
    if k == ord('c'):
        canvas = None
    if k==ord('s') and select_color:
        sb=b
        sr=r
        sg=g
    if k==ord('i') and not select_color:
        if inspect:
            inspect=False
        else:
            inspect=True

cv2.destroyAllWindows()
cap.release()
