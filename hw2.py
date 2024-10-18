from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
import os

lisa_image = cv2.imread('lisa.tif', cv2.IMREAD_GRAYSCALE)
cameraman_image = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
output_path = 'output_images'
if not os.path.exists(output_path):
    os.makedirs(output_path)
plt.imshow(lisa_image, cmap='gray')
plt.title('Lisa Image')
plt.axis('off')
plt.savefig(os.path.join(output_path, 'lisa_image.png'))
plt.close()

plt.imshow(cameraman_image, cmap='gray')
plt.title('Cameraman Image')
plt.axis('off')
plt.savefig(os.path.join(output_path, 'cameraman_image.png'))
plt.close()

plt.hist(lisa_image.ravel(), bins=256, alpha=0.5, label='Lisa Original')
plt.title('Histogram of Lisa')
plt.savefig(os.path.join(output_path, 'histogram_of_lisa.png'))
plt.close()

plt.hist(cameraman_image.ravel(), bins=256, alpha=0.5, label='Cameraman Original')
plt.title('Histogram of Cameraman')
plt.savefig(os.path.join(output_path, 'histogram_of_cameraman.png'))
plt.close()

lisa_matched_image = exposure.match_histograms(lisa_image, cameraman_image)
plt.imshow(lisa_matched_image, cmap='gray')
plt.title('Lisa to Cameraman Matched Image')
plt.axis('off')
plt.savefig(os.path.join(output_path, 'lisa_to_cameraman.png'))
plt.close()

plt.hist(lisa_matched_image.ravel(), bins=256, alpha=0.5, label='Lisa to Cameraman')
plt.title('Lisa Matched Histogram')
plt.savefig(os.path.join(output_path, 'lisa_matched.png'))
plt.close()

cameraman_matched_image = exposure.match_histograms(cameraman_image, lisa_image)
plt.imshow(cameraman_matched_image, cmap='gray')
plt.title('Cameraman to Lisa Matched Image')
plt.axis('off')
plt.savefig(os.path.join(output_path, 'cameraman_to_lisa.png'))
plt.close()

plt.hist(cameraman_matched_image.ravel(), bins=256, alpha=0.5, label='Cameraman to Lisa')
plt.title('Cameraman Matched Histogram')
plt.savefig(os.path.join(output_path, 'cameraman_matched.png'))
plt.close()

lisa_equalized_image = cv2.equalizeHist(lisa_image)
plt.imshow(lisa_equalized_image, cmap='gray')
plt.title('Lisa Equalized Image')
plt.axis('off')
plt.savefig(os.path.join(output_path, 'lisa_equalized.png'))
plt.close()

plt.hist(lisa_equalized_image.ravel(), bins=256, alpha=0.5, label='Lisa Equalized')
plt.title('Lisa Equalized Histogram')
plt.savefig(os.path.join(output_path, 'lisa_equalized_histogram.png'))
plt.close()

cameraman_equalized_image = cv2.equalizeHist(cameraman_image)
plt.imshow(cameraman_equalized_image,cmap='gray')
plt.title('Cameraman Equalized Image')
plt.axis('off')
plt.savefig(os.path.join(output_path, 'cameraman_equalized.png'))
plt.close()

plt.hist(cameraman_equalized_image.ravel(), bins=256, alpha=0.5, label='Cameraman Equalized')
plt.title('Cameraman Equalized Histogram')
plt.savefig(os.path.join(output_path, 'cameraman_equalized_histogram.png'))
plt.close()


plt.figure(figsize=(10, 10))

plt.subplot(3, 2, 1)
plt.imshow(lisa_image, cmap='gray')
plt.title("Original Lisa Image")
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(cameraman_image, cmap='gray')
plt.title("Original Cameraman Image")
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(lisa_matched_image, cmap='gray')
plt.title("Lisa Matched Image")
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(cameraman_matched_image, cmap='gray')
plt.title("Cameraman Matched Image")
plt.axis('off')


# Showing Histograms
plt.subplot(3, 2, 5)
plt.hist(lisa_image.ravel(), bins=256, alpha=0.5, label='Lisa Original')
plt.hist(lisa_matched_image.ravel(), bins=256, alpha=0.5, label='Lisa Matched')
plt.hist(cameraman_image.ravel(), bins=256, alpha=0.5, label='Cameraman Original')
plt.legend(loc='upper right')
plt.title("Histogram Comparison")

plt.subplot(3, 2, 6)
plt.hist(cameraman_image.ravel(), bins=256, alpha=0.5, label='Cameraman Original')
plt.hist(cameraman_matched_image.ravel(), bins=256, alpha=0.5, label='Cameraman Matched')
plt.hist(lisa_image.ravel(), bins=256, alpha=0.5, label='Lisa Original')
plt.legend(loc='upper right')
plt.title("Histogram Comparison")


plt.tight_layout()
plt.show()

