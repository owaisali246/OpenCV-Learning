import math, cv2, datetime
import numpy as np
from matplotlib import pyplot as plt

# ----------------- Image Basics -----------------
def show_image():
    black_bg = np.ones((500, 500), np.uint8) * 0  # for black background
    white_bg = np.ones((500, 500), np.uint8) * 255  # for white background
    cv2.imshow('White Screen',white_bg)
    cv2.imshow('Black Screen',black_bg)

    img1 = cv2.imread("assets/img-2.jpg", 0)  # 0 for grayscale
    img2 = cv2.imread("assets/img-2.jpg", 1)  # 1 for color
    img3 = cv2.imread("assets/img-2.jpg", -1)  # -1 for unchanged

    cv2.imshow("Original Image", img1)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
    elif key == ord("s"):
        cv2.imwrite("assets/img-2-copy.jpg", img1) # to save the image
        cv2.destroyAllWindows()


# --------------- Video Capture ---------------
def start_video_capture():
    cap = cv2.VideoCapture(0)  # 0 for default camera
    # print (cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # to get the width of the frame
    # print (cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # to get the height of the frame
    print (cap.get(cv2.CAP_PROP_FPS)) # to get the fps of the video

    fourcc = cv2.VideoWriter_fourcc(*"XVID") # to get the codec
    # out = cv2.VideoWriter("assets/output.avi", fourcc, 30.0, (640, 480)) # to save the video

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # to set the width of the frame
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # to set the width of the frame
    cap.set(3, 640) # to set the width of the frame
    cap.set(4, 480) # to set the height of the frame

    while (cap.isOpened()):
        ret, frame = cap.read() # ret is boolean, frame is the image
        
        if ret == True:
            # frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY) # for grayscale video
            frame = cv2.flip(frame, 1)
            # flipped = cv2.flip(gray, 1) # flip the image horizontally
            # out.write(frame) # to save the video
            text = "Width: " + str(cap.get(3)) + " Height: " + str(cap.get(4))
            datet = str(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.rectangle(frame, (8, 30), (225, 60), (0, 0, 0), -1) # to draw a rectangle
            frame = cv2.putText(frame, text, (10, 50), font, 0.5, (0, 255, 255), 2) # to put text
            frame = cv2.rectangle(frame, (8, 400), (225, 430), (0, 0, 0), -1) # to draw a rectangle
            frame = cv2.putText(frame, datet, (10, 420), font, 0.5, (0, 255, 255), 2) # to put text
            
            cv2.imshow("Live Video", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        else:
            break
    
    cap.release()
    # out.release() # to save the video
    cv2.destroyAllWindows()


# --------------- Drawing Shapes ---------------
def draw_shapes():
    img = np.ones((500, 500, 3), dtype=np.uint8) * 0
    cv2.line(img, (250,150), (250,350), (0, 255, 255), math.floor(10)) # to draw a line
    cv2.arrowedLine(img, (100,200), (350,200), (205, 230, 0), 10) # to draw an arrowed line
    cv2.rectangle(img, (100, 100), (400, 400), (0, 255, 0), 10) # to draw a rectangle
    cv2.circle(img, (250, 250), 100, (255, 0, 0), -1) # to draw a circle
    cv2.putText(img, "OpenCV", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5) # to put text
    cv2.imshow("Image", img)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

def draw_logo():
    main_radius, mini_radius = 70, 30
    blue, green, red, black = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)
    pt1, pt2, pt3 = (250, 150), (170, 290), (330, 290)
    img = np.ones((500, 500, 3), np.uint8) * 0
    cv2.circle(img, pt1, main_radius, red, -1) # to draw a circle
    cv2.circle(img, pt1, mini_radius, black, -1) # to draw a circle
    cv2.circle(img, pt2, main_radius, green, -1) # to draw a circle
    cv2.circle(img, pt2, mini_radius, black, -1) # to draw a circle
    cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, black, -1)
    cv2.circle(img, pt3, main_radius, blue, -1) # to draw a circle
    cv2.circle(img, pt3, mini_radius, black, -1) # to draw a circle
    cv2.drawContours(img, [np.array([(290,220), (370,220), pt3])], 0, black, -1)
    # cv2.drawContours(img, [np.array([(), (), pt3])], 0, black, -1)
    cv2.putText(img, "OpenCV", (130, 430), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5) # to put text
    cv2.imshow("Line", img)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

def draw_animation():
    for i in range (0, 100):
        img = np.ones((500, 500, 3), np.uint8) * 255
        cv2.line(img, (250-i,150+i), (250+i,350-i), (0, 255-i, 255-i), math.floor(10)) # to draw a line
        cv2.arrowedLine(img, (100,200), (350,200), (205, 30, 0+i), 10) # to draw an arrowed line
        cv2.imshow("Line", img)
        if cv2.waitKey(250) == ord('q'):
            break
    cv2.destroyAllWindows()


# --------------- Mouse Events ---------------
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)

def mouse_events():
    # img = np.ones((500, 500, 3), np.uint8) * 0
    img = cv2.imread("assets/img-2.jpg", 1) 
    cv2.imshow("Image", img)

    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.putText(img, f"({x}, {y})", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        elif event == cv2.EVENT_RBUTTONDOWN:
            blue = img[y, x, 0]
            green = img[y, x, 1]
            red = img[y, x, 2]
            cv2.putText(img, f"({blue}, {green}, {red})", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.setMouseCallback("Image", mouse_event)

    i=0
    while True:
        cv2.imshow("Image", img)
        if cv2.waitKey(250) == ord('q'):
            break
    cv2.destroyAllWindows()

def line_drawing():
    img = np.zeros((500, 500, 3), np.uint8)
    cv2.imshow("Image", img)
    points = []

    def mouse_event(event, x, y, flags, param):
        nonlocal points, img
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x,y), 3, (0, 255, 255), -1)
            points.append((x,y))
            if len(points) >= 2:
                cv2.line(img, points[-1], points[-2], (0, 255, 255), 2)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.clear()
            img[:,:,:] = [0, 0, 0]


    cv2.setMouseCallback("Image", mouse_event)

    while True:
        cv2.imshow("Image", img)
        if cv2.waitKey(250) == ord('q'):
            break
    cv2.destroyAllWindows()

def color_picker():
    img = cv2.imread("assets/img-2.jpg", 1)
    cv2.imshow("Image", img)

    def mouse_event(event, x, y, flags, param) :
        if event == cv2.EVENT_LBUTTONDOWN:
            blue, green, red = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            # cv2.circle(img, (x,y), 1, (0, 255, 255), -1)
            color_window = np.zeros((200, 200, 3), np.uint8)
            color_window[50:,:,:] = [blue, green, red]
            cv2.putText(color_window, f"BGR = ({blue}, {green}, {red})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Color", color_window)

    cv2.setMouseCallback("Image", mouse_event)

    while True:
        cv2.imshow("Image", img)
        if cv2.waitKey(250) == ord('q'):
            break
    cv2.destroyAllWindows()


# --------------- Image Functions ---------------
def image_functions():
    img = cv2.imread("assets/img-2.jpg", 1)
    print(f"Size: {img.shape} \nType: {img.dtype} \nDimensions: {img.ndim} \nShape: {img.shape} \n")
    
    # Splitting and merging channels
    b, g, r = cv2.split(img)
    # img = cv2.merge((r, g, b))
    img2 = cv2.merge([g])

    # ROI = Region of Interest
    # roi = img[239:359, 495:575]
    # img[0:120, 0:80] = roi
    cv2.imshow("Image", img)

    def mouse_event(event, x, y, flags, param):
        nonlocal img
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Pixel: {x, y}")

    cv2.setMouseCallback("Image", mouse_event)

    img3 = cv2.resize(img, (400, 400))
    img4 = cv2.resize(cv2.imread("assets/img-1.jpg", 1), (400, 400))
    # img5 = cv2.add(img4, cv2.subtract(cv2.flip(img4, 1), img4))
    img5 = cv2.addWeighted(img3, 0.5, img4, 0.5, 0)


    while True:
        cv2.imshow("Image", img5)
        if cv2.waitKey(250) == 27 or cv2.waitKey(250) == ord('q'):
            print(chr(27))
            break
    cv2.destroyAllWindows()


# --------------- Trackbars ---------------
def nothing(x):
        pass

def trackbars():
    img = np.zeros((500, 500, 3), np.uint8)
    cv2.namedWindow("Image")

    

    cv2.createTrackbar("B", "Image", 0, 255, nothing)
    cv2.createTrackbar("G", "Image", 0, 255, nothing)
    cv2.createTrackbar("R", "Image", 0, 255, nothing)
    cv2.createTrackbar("Switch", "Image", 0, 1, nothing)

    while True:
        cv2.imshow("Image", img)
        if cv2.waitKey(250) == ord('q'):
            break
        b = cv2.getTrackbarPos("B", "Image")
        g = cv2.getTrackbarPos("G", "Image")
        r = cv2.getTrackbarPos("R", "Image")
        img[:] = [b, g, r]

    cv2.destroyAllWindows()


# --------------- Object Detection and Tracking using HSV Color Space ---------------
def HSVObjectTracker():
    cv2.namedWindow("Tracking")
    cv2.createTrackbar("LH", "Tracking", 0, 255, lambda x: print(x))
    cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UH", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("US", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UV", "Tracking", 0, 255, nothing)

    while True:
        img = cv2.imread("assets/img-2.jpg", 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")
        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow("Image", img)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", res)

        if cv2.waitKey(100) == ord('q'):
            break
    cv2.destroyAllWindows()


# --------------- Image Thresholding ---------------
def imageThreshoulding():
    img = cv2.imread("assets/img-4.jpg", 0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    Titles = ["Original Image", "Binary", "Binary Inverse", "Trunc", "To Zero", "To Zero Inverse"]
    Images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(Images[i], "gray"), plt.title(Titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    # cv2.imshow("Image", img)
    # cv2.imshow("Thresh1", thresh1)
    # cv2.imshow("Thresh2", thresh2)
    # cv2.imshow("Thresh3", thresh3)
    # cv2.imshow("Thresh4", thresh4)
    # cv2.imshow("Thresh5", thresh5)

    # if cv2.waitKey(0) == ord('q'):
    #     cv2.destroyAllWindows()


def adaptiveThresholding():
    img = cv2.imread("assets/img-4.jpg", 0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    cv2.imshow("Image", img)
    cv2.imshow("Thresh1", thresh1)
    cv2.imshow("Thresh2", thresh2)
    cv2.imshow("Thresh3", thresh3)

    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


def matplotWorking():
    img = cv2.imread("assets/img-2.jpg", 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap="gray", interpolation="bicubic")
    plt.xticks([]), plt.yticks([])
    plt.show()


# --------------- Bitwise Operations ---------------
def bitwiseOperations():
    img1 = np.zeros((640, 640, 3), np.uint8)
    img2 = cv2.cvtColor(cv2.imread("assets/img-8.jpg", 1), cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(cv2.imread("assets/img-9.jpg", 1), cv2.COLOR_BGR2RGB)

    bitwiseAnd = cv2.bitwise_and(img3, img2)
    bitwiseOr = cv2.bitwise_or(img2, img3)
    bitwiseXor = cv2.bitwise_xor(img2, img1)
    bitwiseNot = cv2.bitwise_not(img2)

    Titles = ["Image 2", "Image 3", "Bitwise And", "Bitwise Or", "Bitwise Xor", "Bitwise Not"]
    Images = [img2, img3, bitwiseAnd, bitwiseOr, bitwiseXor, bitwiseNot]

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(Images[i], "gray"), plt.title(Titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


