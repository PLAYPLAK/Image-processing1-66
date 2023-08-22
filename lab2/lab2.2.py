import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'lab2/kmitl.jpg'
color_image = cv2.imread(image_path)
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

red, green, blue = cv2.split(color_image)

h_o_r = cv2.calcHist([red], [0], None, [256], [0, 256])
h_o_g = cv2.calcHist([green], [0], None, [256], [0, 256])
h_o_b = cv2.calcHist([blue], [0], None, [256], [0, 256])
r_eq = cv2.equalizeHist(red)
g_eq = cv2.equalizeHist(green)
b_eq = cv2.equalizeHist(blue)
h_eq_r = cv2.calcHist([r_eq], [0], None, [256], [0, 256])
h_eq_g = cv2.calcHist([g_eq], [0], None, [256], [0, 256])
h_eq_b = cv2.calcHist([b_eq], [0], None, [256], [0, 256])
plt.figure(figsize=(12, 10))


plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(h_o_r, color='red', label='Red')
plt.plot(h_o_g, color='green', label='Green')
plt.plot(h_o_b, color='blue', label='Blue')
plt.title('Original Image Histograms')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()


plt.subplot(2, 2, 3)
equalized_color_image = cv2.merge((b_eq, g_eq, r_eq))
plt.imshow(cv2.cvtColor(equalized_color_image, cv2.COLOR_BGR2RGB))
plt.title('Equalized Image')
plt.axis('off')


plt.subplot(2, 2, 4)
plt.plot(h_eq_r, color='red', label='Red')
plt.plot(h_eq_g, color='green', label='Green')
plt.plot(h_eq_b, color='blue', label='Blue')
plt.title('Equalized Image Histograms')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()