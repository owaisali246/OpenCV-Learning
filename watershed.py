import cv2 as cv
import numpy as np

def algo_2():

    # Read the input image
    image = cv.imread('assets/img-4.jpg')
    original = image.copy()

    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Noise removal using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area using the distance transform
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply Watershed algorithm
    cv.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]  # Mark watershed boundary with red color

    # Find contours in the watershed result
    contours, _ = cv.findContours(markers.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    for i in range(len(contours)):
        cv.drawContours(original, contours, i, (0, 255, 0), 2)

    # Display the results
    cv.imshow('Original Image', original)
    cv.imshow('Watershed Result', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def algo_1():

    img = cv.imread('assets/img-4.jpg')
    img2 = cv.imread('assets/img-4.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    # Display the results
    cv.imshow('Original Image', img)
    cv.imshow('Watershed Result', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

algo_2()