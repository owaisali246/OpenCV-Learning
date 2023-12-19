import cv2 as cv
import numpy as np

# Read the input image
img1 = cv.imread('assets/img-5.jpg') # Refference image
img2 = cv.imread('assets/img-6.jpg') # Image to be aligned

# Convert the image to grayscale
gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors.
MAX_FEATURES = 500
orb = cv.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(gray_img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_img2, None)

# Display ORB keypoints
img1_keypoints = cv.drawKeypoints(img1, keypoints1, None)
img2_keypoints = cv.drawKeypoints(img2, keypoints2, None)
cv.imshow('ORB keypoints 1', img1_keypoints)
cv.imshow('ORB keypoints 2', img2_keypoints)
cv.waitKey()

# Match features.
matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# Sort matches by score
matches = list(matches)
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * 0.15)
matches = matches[:numGoodMatches]

# Draw top matches
img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
cv.imshow('Matches', img_matches)
cv.waitKey()

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv.findHomography(points2, points1, cv.RANSAC)

# Use homography
height, width, channels = img1.shape
img2Reg = cv.warpPerspective(img2, h, (width, height))

# Show final results
cv.imshow('Original Image', img2)
cv.imshow('Registered Image', img2Reg)
cv.waitKey()
cv.destroyAllWindows()




