import math, cv2, datetime
import numpy as np
from pyparsing import col

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

