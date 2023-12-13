import cv2
import numpy as np
from matplotlib import pyplot as plt

def LaplacianFilter():
    # Read the image
    image = cv2.imread('assets/img-1.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Convert the result to 8-bit for display
    laplacian = np.uint8(np.absolute(laplacian))

    # Display the original and the Laplacian-filtered images
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian Filter'), plt.xticks([]), plt.yticks([])

    plt.show()


def SobelFilter():
    # Read the image
    image = cv2.imread('assets/img-2.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply Sobel filter in the x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Convert the results to absolute values
    sobel_x = np.abs(sobel_x)
    sobel_y = np.abs(sobel_y)

    # Combine the results to get the final edge-detected image
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Convert the result to 8-bit for display
    sobel_combined = np.uint8(sobel_combined)

    # Display the original and the Sobel-filtered images
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Filter'), plt.xticks([]), plt.yticks([])

    plt.show()
