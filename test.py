# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# image = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
# c = 255 / np.log(1 + np.max(image))

# gamma = 1.2  # Gama 
# gamma_transformed = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)


# negative_image = 255 - image

# log_transformed = c * (np.log(1+image))

# log_transformed = np.array(log_transformed, dtype=np.uint8)


# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title("Original Image")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(gamma_transformed, cmap='gray')
# plt.title("Gamma Transformed Image")
# plt.axis('off')

# plt.show()

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.hist(image.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.5, label='Orijinal')
# plt.title("Original Histogram")

# plt.subplot(1, 2, 2)
# plt.hist(gamma_transformed.ravel(), bins=256, range=(0, 255), color='green', alpha=0.5, label='Log Dönüşümlü')
# plt.title("Gamma Transformed Histogram")

# plt.show()






# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Kontrast iyileştirme teknikleri için fonksiyonlar

# # 1. Logaritmik Dönüşüm
# def log_transform(image):
#     c = 255 / np.log(1 + np.max(image))
#     log_image = c * (np.log(1 + image))
#     return np.array(log_image, dtype=np.uint8)

# # 2. Negatif Dönüşüm
# def negative_transform(image):
#     return 255 - image

# # 3. Gama Dönüşümü
# def gamma_transform(image, gamma):
#     gamma_image = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
#     return gamma_image

# # 4. Parça-Parça Doğrusal Dönüşüm (Piecewise Linear Transformation)
# def piecewise_linear_transform(image):
#     # Basit bir doğrusal dönüşüm örneği
#     r1, s1 = 70, 0
#     r2, s2 = 140, 255

#     def transformation_function(x):
#         if x < r1:
#             return s1 / r1 * x
#         elif x < r2:
#             return ((s2 - s1) / (r2 - r1)) * (x - r1) + s1
#         else:
#             return ((255 - s2) / (255 - r2)) * (x - r2) + s2

#     vectorized_transformation = np.vectorize(transformation_function)
#     return vectorized_transformation(image).astype(np.uint8)

# # Görüntü ve histogram gösterme fonksiyonu
# def show_image_and_histogram(image, title):
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title(title)
#     plt.axis('off')

#     # Histogram gösterimi
#     plt.subplot(1, 2, 2)
#     plt.hist(image.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.5)
#     plt.title(f"{title} Histogram")
#     plt.show()

# # Ana fonksiyon: Dönüşüm seçme ve uygulama
# def apply_contrast_enhancement(image, method, gamma=None):
#     if method == 'log':
#         return log_transform(image)
#     elif method == 'negative':
#         return negative_transform(image)
#     elif method == 'gamma':
#         if gamma is None:
#             raise ValueError("Gama dönüşümü için bir gamma değeri belirtmelisiniz.")
#         return gamma_transform(image, gamma)
#     elif method == 'piecewise_linear':
#         return piecewise_linear_transform(image)
#     else:
#         raise ValueError(f"Bilinmeyen method: {method}")

# # Kullanmaya başla: Lisa resmini yükle
# lisa_image = cv2.imread('lisa.tif', cv2.IMREAD_GRAYSCALE)
# cameraman_image = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
# # Uygulanacak yöntem
# method = 'piecewise_linear'  # 'log', 'negative', 'gamma', 'piecewise_linear' seçenekleri var
# image = lisa_image
# # Gama için bir değer belirleyin (gama dönüşümü uygulanacaksa)
# gamma_value = 2.0

# # Dönüşümü uygula
# if method == 'gamma':
#     transformed_image = apply_contrast_enhancement(image, method, gamma=gamma_value)
# else:
#     transformed_image = apply_contrast_enhancement(image, method)

# # Orijinal ve dönüşümlü görüntüyü ve histogramlarını göster
# show_image_and_histogram(lisa_image, f"Original {image}")
# show_image_and_histogram(transformed_image, f"{method.capitalize()} Dönüştürülmüş Lisa Görüntüsü")







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
