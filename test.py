import cv2
import numpy as np
import matplotlib.pyplot as plt

# ImageObject class
class ImageObject:
    def __init__(self, image_path, name):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.name = name
        if self.image is None:
            raise ValueError(f"There is no image: {image_path}")

    def show_image_and_histogram(self, transformed_image=None, title=None):
        # if there is no transformed image show the original one
        image_to_show = transformed_image if transformed_image is not None else self.image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_to_show, cmap='gray')
        plt.title(title or self.name)
        plt.axis('off')

        # Showing Histogram
        plt.subplot(1, 2, 2)
        plt.hist(image_to_show.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.5)
        plt.title(f"{self.name} Histogram")
        plt.show()

    def apply_contrast_enhancement(self, method, gamma=None):
        if method == 'log':
            return self.log_transform()
        elif method == 'negative':
            return self.negative_transform()
        elif method == 'gamma':
            if gamma is None:
                raise ValueError("You should add a value for gamma transformation.") # this function provide enhancement techniques
            return self.gamma_transform(gamma)
        elif method == 'piecewise_linear':
            return self.piecewise_linear_transform()
        else:
            raise ValueError(f"Unknown: {method}")

    def log_transform(self):
        c = 255 / np.log(1 + np.max(self.image))
        log_image = c * (np.log(1 + self.image))
        return np.array(log_image, dtype=np.uint8)

    def negative_transform(self):
        return 255 - self.image

    def gamma_transform(self, gamma):
        gamma_image = np.array(255 * (self.image / 255) ** gamma, dtype=np.uint8)
        return gamma_image

    def piecewise_linear_transform(self):
        r1, s1 = 70, 0
        r2, s2 = 140, 255

        def transformation_function(x):
            if x < r1:
                return s1 / r1 * x
            elif x < r2:
                return ((s2 - s1) / (r2 - r1)) * (x - r1) + s1
            else:
                return ((255 - s2) / (255 - r2)) * (x - r2) + s2

        vectorized_transformation = np.vectorize(transformation_function)
        return vectorized_transformation(self.image).astype(np.uint8)


# we create lisa and cameraman objects
lisa = ImageObject('lisa.tif', 'Lisa')
cameraman = ImageObject('cameraman.tif', 'Cameraman')

# user choose the which images
image_to_use = lisa  # it can be 'lisa' or 'cameraman' 

# method which be implement
method = 'piecewise_linear'  # there are 4 options 'log', 'negative', 'gamma', 'piecewise_linear' 
gamma_value = 0.5  # gamma value

# gamma transformation
if method == 'gamma':
    transformed_image = image_to_use.apply_contrast_enhancement(method, gamma=gamma_value)
else:
    transformed_image = image_to_use.apply_contrast_enhancement(method)

# I showe the original and transformed images with their histograms
image_to_use.show_image_and_histogram(title="Original Image")
image_to_use.show_image_and_histogram(transformed_image, title=f"{method.capitalize()} Transformed Image")
